"""PostgreSQL database backend for Datasette.

Uses psycopg v3 in synchronous mode within Datasette's thread pool.
Key differences from SQLite:
- No write queue needed (PostgreSQL uses MVCC for concurrent writes)
- Read connections use default_transaction_read_only=on
- Statement timeouts use PostgreSQL's statement_timeout setting
- Schema introspection via information_schema / pg_catalog
- Parameter binding uses %(name)s style instead of :name
"""

import asyncio
import re
import sys
from collections import Counter
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg
import psycopg.errors
import psycopg.rows

from .base import DatabaseBackend, Column
from ..tracer import trace


# PostgreSQL reserved words (common ones that need quoting)
_pg_reserved_words = set(
    (
        "all analyse analyze and any array as asc asymmetric authorization between "
        "binary both case cast check collate collation column concurrently constraint "
        "create cross current_catalog current_date current_role current_schema "
        "current_time current_timestamp current_user default deferrable desc distinct "
        "do else end except false fetch for foreign freeze from full grant group having "
        "ilike in initially inner intersect into is isnull join lateral leading left "
        "like limit localtime localtimestamp natural not notnull null offset on only or "
        "order outer overlaps placing primary references returning right select session_user "
        "similar some symmetric table tablesample then to trailing true union unique user "
        "using variadic verbose when where window with"
    ).split()
)

_boring_keyword_re = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Regex to convert :name params to %(name)s, avoiding :: casts
_param_re = re.compile(r"(?<!:):([a-zA-Z_]\w*)")


class PostgresRow:
    """Row class satisfying RowProtocol for psycopg results.

    Supports integer indexing, string key access, iteration, len, and keys().
    """

    __slots__ = ("_values", "_columns", "_mapping")

    def __init__(self, cursor):
        """Row factory for psycopg - called with cursor to get a "maker",
        then the maker is called with each row's values."""
        # This is actually the "maker" factory pattern psycopg expects
        pass

    @classmethod
    def row_factory(cls, cursor):
        """psycopg row factory: returns a callable that creates rows."""
        if cursor.description is None:
            # DDL/DML commands without result set
            return lambda values: values
        columns = [desc.name for desc in cursor.description]

        def make_row(values):
            row = object.__new__(cls)
            row._values = tuple(values)
            row._columns = columns
            row._mapping = dict(zip(columns, row._values))
            return row

        return make_row

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._columns


class PostgresBackend(DatabaseBackend):
    backend_type = "postgresql"

    def __init__(
        self,
        ds=None,
        connection_string=None,
        schema=None,
        statement_timeout_ms=100,
    ):
        self.ds = ds
        self.connection_string = connection_string
        self.schema = schema
        self.statement_timeout_ms = statement_timeout_ms

        # Connection tracking
        self._all_connections = []

    # ---- Connection lifecycle ----

    def create_connection(self, write=False):
        options_parts = []
        if not write:
            options_parts.append("-c default_transaction_read_only=on")
        options_parts.append(f"-c statement_timeout={self.statement_timeout_ms}")
        if self.schema:
            options_parts.append(f"-c search_path={self.schema}")
        options = " ".join(options_parts)

        conn = psycopg.connect(
            self.connection_string,
            options=options,
            autocommit=True,
            row_factory=PostgresRow.row_factory,
        )
        self._all_connections.append(conn)
        return conn

    def close_connection(self, conn):
        conn.close()
        try:
            self._all_connections.remove(conn)
        except ValueError:
            pass

    def close_all(self):
        for conn in self._all_connections:
            try:
                conn.close()
            except Exception:
                pass
        self._all_connections.clear()

    def prepare_connection(self, conn, datasette, database_name):
        # PostgreSQL connections are configured at creation time via options
        # No additional preparation needed
        pass

    # ---- Async execution ----

    async def execute(
        self,
        sql,
        params=None,
        truncate=False,
        custom_time_limit=None,
        page_size=None,
        log_sql_errors=True,
    ):
        page_size = page_size or (self.ds.page_size if self.ds else 50)
        translated_sql = self.translate_sql(sql)

        def sql_operation_in_thread(conn):
            time_limit_ms = self.ds.sql_time_limit_ms if self.ds else self.statement_timeout_ms
            if custom_time_limit and custom_time_limit < time_limit_ms:
                time_limit_ms = custom_time_limit

            with self.time_limit_context(conn, time_limit_ms):
                try:
                    cursor = conn.execute(
                        translated_sql, params if params is not None else {}
                    )
                    max_returned_rows = (
                        self.ds.max_returned_rows if self.ds else 100
                    )
                    if max_returned_rows == page_size:
                        max_returned_rows += 1
                    if max_returned_rows and truncate:
                        rows = cursor.fetchmany(max_returned_rows + 1)
                        truncated = len(rows) > max_returned_rows
                        rows = rows[:max_returned_rows]
                    else:
                        rows = cursor.fetchall()
                        truncated = False
                except psycopg.errors.QueryCanceled as e:
                    from ..database import QueryInterrupted

                    raise QueryInterrupted(e, sql, params)
                except (psycopg.errors.OperationalError, psycopg.errors.DatabaseError) as e:
                    if log_sql_errors:
                        sys.stderr.write(
                            "ERROR: conn={}, sql = {}, params = {}: {}\n".format(
                                conn, repr(sql), params, e
                            )
                        )
                        sys.stderr.flush()
                    raise

            from ..database import Results

            if truncate:
                return Results(rows, truncated, cursor.description)
            else:
                return Results(rows, False, cursor.description)

        with trace(
            "sql",
            database=getattr(self, "_database_name", ""),
            sql=sql.strip(),
            params=params,
        ):
            results = await self.execute_fn(sql_operation_in_thread)
        return results

    async def execute_fn(self, fn):
        if self.ds is None or self.ds.executor is None:
            # non-threaded mode
            conn = self.create_connection()
            try:
                return fn(conn)
            finally:
                self.close_connection(conn)

        # threaded mode - use a fresh connection per call
        # (psycopg connections are not thread-safe)
        def in_thread():
            conn = self.create_connection()
            try:
                return fn(conn)
            finally:
                self.close_connection(conn)

        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )

    async def execute_write_fn(
        self, fn, block=True, transaction=True, request=None
    ):
        """Execute fn(conn) using a write connection.

        PostgreSQL doesn't need a write queue - MVCC handles concurrent writes.
        """
        fn = self._wrap_fn_with_hooks(fn, request, transaction)

        if self.ds is None or self.ds.executor is None:
            # non-threaded mode
            conn = self.create_connection(write=True)
            try:
                if transaction:
                    with conn.transaction():
                        return fn(conn)
                else:
                    return fn(conn)
            finally:
                self.close_connection(conn)

        # threaded mode
        def in_thread():
            conn = self.create_connection(write=True)
            try:
                if transaction:
                    with conn.transaction():
                        return fn(conn)
                else:
                    return fn(conn)
            finally:
                self.close_connection(conn)

        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )

    async def execute_write(self, sql, params=None, block=True, request=None):
        translated_sql = self.translate_sql(sql)

        def _inner(conn):
            return conn.execute(translated_sql, params or {})

        with trace(
            "sql",
            database=getattr(self, "_database_name", ""),
            sql=sql.strip(),
            params=params,
        ):
            results = await self.execute_write_fn(
                _inner, block=block, request=request
            )
        return results

    async def execute_write_script(self, sql, block=True, request=None):
        """Execute multiple statements separated by semicolons."""

        def _inner(conn):
            # psycopg can execute multiple statements
            return conn.execute(sql)

        with trace(
            "sql",
            database=getattr(self, "_database_name", ""),
            sql=sql.strip(),
            executescript=True,
        ):
            results = await self.execute_write_fn(
                _inner, block=block, transaction=False, request=request
            )
        return results

    async def execute_write_many(self, sql, params_seq, block=True, request=None):
        translated_sql = self.translate_sql(sql)

        def _inner(conn):
            count = 0
            cursor = conn.cursor()
            for params in params_seq:
                cursor.execute(translated_sql, params)
                count += 1
            return cursor, count

        with trace(
            "sql",
            database=getattr(self, "_database_name", ""),
            sql=sql.strip(),
            executemany=True,
        ) as kwargs:
            results, count = await self.execute_write_fn(
                _inner, block=block, request=request
            )
            kwargs["count"] = count
        return results

    async def execute_isolated_fn(self, fn):
        """Execute fn on a dedicated connection."""
        if self.ds is None or self.ds.executor is None:
            conn = self.create_connection(write=True)
            try:
                return fn(conn)
            finally:
                self.close_connection(conn)

        def in_thread():
            conn = self.create_connection(write=True)
            try:
                return fn(conn)
            finally:
                self.close_connection(conn)

        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )

    def _wrap_fn_with_hooks(self, fn, request, transaction):
        if self.ds is None:
            return fn
        from ..plugins import pm

        wrappers = pm.hook.write_wrapper(
            datasette=self.ds,
            database=getattr(self, "_database_name", ""),
            request=request,
            transaction=transaction,
        )
        wrappers = [w for w in wrappers if w is not None]
        if not wrappers:
            return fn
        original_fn = fn
        for wrapper_factory in reversed(wrappers):
            original_fn = _apply_write_wrapper(original_fn, wrapper_factory)
        return original_fn

    # ---- SQL dialect ----

    def translate_sql(self, sql):
        """Convert :name params to %(name)s for psycopg, preserving :: casts."""
        return _param_re.sub(r"%(\1)s", sql)

    def escape_identifier(self, identifier):
        if _boring_keyword_re.match(identifier) and (
            identifier.lower() not in _pg_reserved_words
        ):
            return identifier
        else:
            # Use double quotes for PostgreSQL (standard SQL)
            escaped = identifier.replace('"', '""')
            return f'"{escaped}"'

    # ---- Time limiting ----

    @contextmanager
    def time_limit_context(self, conn, ms):
        """Set per-query statement_timeout using SET LOCAL."""
        # For PostgreSQL, we need a transaction for SET LOCAL
        # Since we use autocommit, we use a subtransaction
        try:
            conn.execute(f"SET statement_timeout = {int(ms)}")
            yield
        finally:
            conn.execute(f"SET statement_timeout = {self.statement_timeout_ms}")

    def is_interrupted_error(self, error):
        return isinstance(error, psycopg.errors.QueryCanceled)

    def is_operational_error(self, error):
        return isinstance(
            error, (psycopg.errors.OperationalError, psycopg.errors.DatabaseError)
        )

    # ---- Schema introspection ----

    def table_names(self, conn):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %(schema)s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            {"schema": schema},
        ).fetchall()
        return [r[0] for r in rows]

    def view_names(self, conn):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = %(schema)s
            ORDER BY table_name
            """,
            {"schema": schema},
        ).fetchall()
        return [r[0] for r in rows]

    def table_exists(self, conn, table):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %(schema)s
              AND table_name = %(table)s
              AND table_type = 'BASE TABLE'
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        return bool(rows)

    def view_exists(self, conn, view):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = %(schema)s
              AND table_name = %(view)s
            """,
            {"schema": schema, "view": view},
        ).fetchall()
        return bool(rows)

    def table_columns(self, conn, table):
        return [col.name for col in self.table_column_details(conn, table)]

    def table_column_details(self, conn, table):
        schema = self.schema or "public"
        # Get columns with their ordinal position
        rows = conn.execute(
            """
            SELECT
                c.ordinal_position - 1 AS cid,
                c.column_name AS name,
                c.data_type AS type,
                CASE WHEN c.is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                c.column_default AS default_value,
                CASE WHEN pk.column_name IS NOT NULL THEN pk.ordinal_position ELSE 0 END AS is_pk,
                0 AS hidden
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.column_name, kcu.ordinal_position
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = %(schema)s
                  AND tc.table_name = %(table)s
                  AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_schema = %(schema)s
              AND c.table_name = %(table)s
            ORDER BY c.ordinal_position
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        return [Column(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    def primary_keys(self, conn, table):
        columns = self.table_column_details(conn, table)
        pks = [col for col in columns if col.is_pk]
        pks.sort(key=lambda col: col.is_pk)
        return [col.name for col in pks]

    def foreign_keys_for_table(self, conn, table):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT
                kcu.column_name AS from_column,
                ccu.table_name AS to_table,
                ccu.column_name AS to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
              AND tc.table_schema = ccu.table_schema
            WHERE tc.table_schema = %(schema)s
              AND tc.table_name = %(table)s
              AND tc.constraint_type = 'FOREIGN KEY'
            ORDER BY kcu.ordinal_position
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        return [
            {
                "column": r[0],
                "other_table": r[1],
                "other_column": r[2],
            }
            for r in rows
        ]

    def get_all_foreign_keys(self, conn):
        tables = self.table_names(conn)
        table_to_foreign_keys = {}
        for table in tables:
            table_to_foreign_keys[table] = {"incoming": [], "outgoing": []}
        for table in tables:
            fks = self.foreign_keys_for_table(conn, table)
            for fk in fks:
                table_name = fk["other_table"]
                from_ = fk["column"]
                to_ = fk["other_column"]
                if table_name not in table_to_foreign_keys:
                    continue
                table_to_foreign_keys[table_name]["incoming"].append(
                    {"other_table": table, "column": to_, "other_column": from_}
                )
                table_to_foreign_keys[table]["outgoing"].append(
                    {"other_table": table_name, "column": from_, "other_column": to_}
                )
        for table in table_to_foreign_keys:
            table_to_foreign_keys[table]["incoming"].sort(
                key=lambda fk: (fk["other_table"], fk["column"], fk["other_column"])
            )
            table_to_foreign_keys[table]["outgoing"].sort(
                key=lambda fk: (fk["other_table"], fk["column"], fk["other_column"])
            )
        return table_to_foreign_keys

    def hidden_table_names(self, conn):
        # PostgreSQL has no shadow tables like SQLite
        return []

    def get_table_definition(self, conn, table, type_="table"):
        schema = self.schema or "public"
        if type_ == "view":
            return self.get_view_definition(conn, table)

        # Build a CREATE TABLE approximation from information_schema
        columns = self.table_column_details(conn, table)
        if not columns:
            return None

        pk_cols = [c.name for c in columns if c.is_pk]
        col_defs = []
        for col in columns:
            parts = [self.escape_identifier(col.name), col.type]
            if col.notnull:
                parts.append("NOT NULL")
            if col.default_value is not None:
                parts.append(f"DEFAULT {col.default_value}")
            col_defs.append(" ".join(parts))

        if pk_cols:
            pk_str = ", ".join(self.escape_identifier(c) for c in pk_cols)
            col_defs.append(f"PRIMARY KEY ({pk_str})")

        return "CREATE TABLE {} (\n  {}\n);".format(
            self.escape_identifier(table),
            ",\n  ".join(col_defs),
        )

    def get_view_definition(self, conn, view):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT view_definition
            FROM information_schema.views
            WHERE table_schema = %(schema)s
              AND table_name = %(view)s
            """,
            {"schema": schema, "view": view},
        ).fetchall()
        if not rows:
            return None
        return "CREATE VIEW {} AS\n{}".format(
            self.escape_identifier(view), rows[0][0]
        )

    def indexes_for_table(self, conn, table):
        schema = self.schema or "public"
        rows = conn.execute(
            """
            SELECT
                i.relname AS name,
                ix.indisunique AS unique,
                pg_get_indexdef(ix.indexrelid) AS sql
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = %(schema)s
              AND t.relname = %(table)s
            ORDER BY i.relname
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        return [{"name": r[0], "unique": r[1], "sql": r[2]} for r in rows]

    def detect_fts(self, conn, table):
        # PostgreSQL doesn't use SQLite-style FTS tables
        return None

    def supports_fts(self):
        return False

    def schema_version(self, conn):
        """Use pg_catalog to detect schema changes via a checksum of table/column info."""
        schema = self.schema or "public"
        row = conn.execute(
            """
            SELECT md5(string_agg(
                table_name || '.' || column_name || '.' || data_type,
                ',' ORDER BY table_name, ordinal_position
            )) AS hash
            FROM information_schema.columns
            WHERE table_schema = %(schema)s
            """,
            {"schema": schema},
        ).fetchone()
        if row and row[0]:
            # Convert first 8 hex chars to int for a comparable version number
            return int(row[0][:8], 16)
        return 0

    def suggest_name(self):
        """Extract database name from the connection string."""
        if self.connection_string:
            parsed = urlparse(self.connection_string)
            db_name = parsed.path.lstrip("/")
            if db_name:
                return db_name
        return "db"


def _apply_write_wrapper(fn, wrapper_factory):
    def wrapped(conn):
        gen = wrapper_factory(conn)
        try:
            next(gen)
        except StopIteration:
            return fn(conn)
        try:
            result = fn(conn)
        except Exception:
            try:
                gen.throw(*sys.exc_info())
            except StopIteration:
                pass
            raise
        else:
            try:
                gen.send(result)
            except StopIteration:
                pass
            return result

    return wrapped
