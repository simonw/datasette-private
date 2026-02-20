"""PostgreSQL database backend for Datasette.

Uses psycopg v3 async support for non-blocking database operations.
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
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import psycopg
import psycopg.errors
import psycopg.rows

from datasette.backends.base import DatabaseBackend, Column
from datasette.tracer import trace


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

    # ---- Connection lifecycle ----

    def _connection_options(self, write=False):
        options_parts = []
        if not write:
            options_parts.append("-c default_transaction_read_only=on")
        options_parts.append(f"-c statement_timeout={self.statement_timeout_ms}")
        if self.schema:
            options_parts.append(f"-c search_path={self.schema}")
        return " ".join(options_parts)

    async def _get_read_conn(self):
        """Get or create a persistent async read connection."""
        if (
            not hasattr(self, "_read_conn")
            or self._read_conn is None
            or self._read_conn.closed
        ):
            self._read_conn = await psycopg.AsyncConnection.connect(
                self.connection_string,
                options=self._connection_options(write=False),
                autocommit=True,
                row_factory=PostgresRow.row_factory,
            )
        return self._read_conn

    @asynccontextmanager
    async def _async_conn(self, write=False):
        if not write:
            yield await self._get_read_conn()
        else:
            conn = await psycopg.AsyncConnection.connect(
                self.connection_string,
                options=self._connection_options(write=True),
                autocommit=True,
                row_factory=PostgresRow.row_factory,
            )
            try:
                yield conn
            finally:
                await conn.close()

    def _sync_conn(self, write=False):
        return psycopg.connect(
            self.connection_string,
            options=self._connection_options(write),
            autocommit=True,
            row_factory=PostgresRow.row_factory,
        )

    def create_connection(self, write=False):
        return self._sync_conn(write)

    def close_connection(self, conn):
        conn.close()

    def close_all(self):
        if hasattr(self, "_read_conn") and self._read_conn and not self._read_conn.closed:
            # Schedule async close if event loop is running
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._read_conn.close())
                else:
                    loop.run_until_complete(self._read_conn.close())
            except Exception:
                pass
            self._read_conn = None

    def prepare_connection(self, conn, datasette, database_name):
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

        if isinstance(params, (list, tuple)):
            pg_params = tuple(params)
        elif params is not None:
            pg_params = params
        else:
            pg_params = {}

        time_limit_ms = self.ds.sql_time_limit_ms if self.ds else self.statement_timeout_ms
        if custom_time_limit and custom_time_limit < time_limit_ms:
            time_limit_ms = custom_time_limit

        with trace(
            "sql",
            database=getattr(self, "_database_name", ""),
            sql=sql.strip(),
            params=params,
        ):
            async with self._async_conn() as conn:
                async with self._async_time_limit(conn, time_limit_ms):
                    try:
                        cursor = await conn.execute(translated_sql, pg_params)
                        max_returned_rows = (
                            self.ds.max_returned_rows if self.ds else 100
                        )
                        if max_returned_rows == page_size:
                            max_returned_rows += 1
                        if max_returned_rows and truncate:
                            rows = await cursor.fetchmany(max_returned_rows + 1)
                            truncated = len(rows) > max_returned_rows
                            rows = rows[:max_returned_rows]
                        else:
                            rows = await cursor.fetchall()
                            truncated = False
                    except psycopg.errors.QueryCanceled as e:
                        from datasette.database import QueryInterrupted

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

        from datasette.database import Results

        if truncate:
            return Results(rows, truncated, cursor.description)
        else:
            return Results(rows, False, cursor.description)

    async def execute_fn(self, fn):
        """Run fn(conn) using an async connection.

        For schema introspection callbacks that do sync conn.execute() calls,
        we use asyncio.to_thread with a sync connection.
        """
        return await asyncio.to_thread(self._run_fn_sync, fn)

    def _run_fn_sync(self, fn, write=False):
        conn = self._sync_conn(write)
        try:
            return fn(conn)
        finally:
            conn.close()

    async def execute_write_fn(
        self, fn, block=True, transaction=True, request=None
    ):
        """Execute fn(conn) using a write connection.

        PostgreSQL doesn't need a write queue - MVCC handles concurrent writes.
        """
        fn = self._wrap_fn_with_hooks(fn, request, transaction)

        if transaction:
            def fn_with_txn(conn):
                with conn.transaction():
                    return fn(conn)
            return await asyncio.to_thread(self._run_fn_sync, fn_with_txn, True)
        else:
            return await asyncio.to_thread(self._run_fn_sync, fn, True)

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
        return await asyncio.to_thread(self._run_fn_sync, fn, True)

    def _wrap_fn_with_hooks(self, fn, request, transaction):
        if self.ds is None:
            return fn
        from datasette.plugins import pm

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
        """Convert :name params to %(name)s and ? to %s for psycopg, preserving :: casts."""
        # First convert :name to %(name)s
        sql = _param_re.sub(r"%(\1)s", sql)
        # Then convert ? positional placeholders to %s
        # Be careful not to convert ?? (which would be a literal ?)
        sql = re.sub(r"(?<!\?)\?(?!\?)", "%s", sql)
        return sql

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

    @asynccontextmanager
    async def _async_time_limit(self, conn, ms):
        """Set per-query statement_timeout using async connection."""
        try:
            await conn.execute(f"SET statement_timeout = {int(ms)}")
            yield
        finally:
            await conn.execute(f"SET statement_timeout = {self.statement_timeout_ms}")

    def time_limit_context(self, conn, ms):
        """Sync time limit context - only used by execute_fn callbacks."""
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            try:
                conn.execute(f"SET statement_timeout = {int(ms)}")
                yield
            finally:
                conn.execute(f"SET statement_timeout = {self.statement_timeout_ms}")

        return ctx()

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

    def label_column_details(self, conn, table):
        schema = self.schema or "public"
        # Get column names and types
        col_rows = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %(schema)s
              AND table_name = %(table)s
            ORDER BY ordinal_position
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        # Find unique single-column indexes
        unique_cols = set()
        idx_rows = conn.execute(
            """
            SELECT a.attname
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            WHERE n.nspname = %(schema)s
              AND t.relname = %(table)s
              AND ix.indisunique = true
              AND ix.indnatts = 1
            """,
            {"schema": schema, "table": table},
        ).fetchall()
        for r in idx_rows:
            unique_cols.add(r[0])
        # Map PG types to Python types for label detection
        _text_types = {
            "text", "character varying", "varchar", "char", "character",
            "name", "citext",
        }
        details = {}
        for col_name, data_type in col_rows:
            py_type = str if data_type in _text_types else type(None)
            details[col_name] = (py_type, col_name in unique_cols)
        return details

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

    # ---- Async schema introspection ----
    # These bypass execute_fn/threads and use async connections directly.

    async def _async_fetch(self, sql, params=None):
        async with self._async_conn() as conn:
            cursor = await conn.execute(sql, params or {})
            return await cursor.fetchall()

    async def async_table_names(self):
        schema = self.schema or "public"
        rows = await self._async_fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %(schema)s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            {"schema": schema},
        )
        return [r[0] for r in rows]

    async def async_view_names(self):
        schema = self.schema or "public"
        rows = await self._async_fetch(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = %(schema)s
            ORDER BY table_name
            """,
            {"schema": schema},
        )
        return [r[0] for r in rows]

    async def async_table_exists(self, table):
        schema = self.schema or "public"
        rows = await self._async_fetch(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %(schema)s
              AND table_name = %(table)s
              AND table_type = 'BASE TABLE'
            """,
            {"schema": schema, "table": table},
        )
        return bool(rows)

    async def async_view_exists(self, view):
        schema = self.schema or "public"
        rows = await self._async_fetch(
            """
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = %(schema)s
              AND table_name = %(view)s
            """,
            {"schema": schema, "view": view},
        )
        return bool(rows)

    async def async_table_column_details(self, table):
        schema = self.schema or "public"
        rows = await self._async_fetch(
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
        )
        return [Column(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    async def async_table_columns(self, table):
        return [col.name for col in await self.async_table_column_details(table)]

    async def async_primary_keys(self, table):
        columns = await self.async_table_column_details(table)
        pks = [col for col in columns if col.is_pk]
        pks.sort(key=lambda col: col.is_pk)
        return [col.name for col in pks]

    async def async_detect_fts(self, table):
        return None

    async def async_foreign_keys_for_table(self, table):
        schema = self.schema or "public"
        rows = await self._async_fetch(
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
        )
        return [
            {"column": r[0], "other_table": r[1], "other_column": r[2]}
            for r in rows
        ]

    async def async_get_all_foreign_keys(self):
        tables = await self.async_table_names()
        table_to_foreign_keys = {}
        for table in tables:
            table_to_foreign_keys[table] = {"incoming": [], "outgoing": []}
        for table in tables:
            fks = await self.async_foreign_keys_for_table(table)
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

    async def async_hidden_table_names(self):
        return []

    async def async_get_table_definition(self, table, type_="table"):
        if type_ == "view":
            return await self.async_get_view_definition(table)
        columns = await self.async_table_column_details(table)
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
            self.escape_identifier(table), ",\n  ".join(col_defs),
        )

    async def async_get_view_definition(self, view):
        schema = self.schema or "public"
        rows = await self._async_fetch(
            """
            SELECT view_definition
            FROM information_schema.views
            WHERE table_schema = %(schema)s
              AND table_name = %(view)s
            """,
            {"schema": schema, "view": view},
        )
        if not rows:
            return None
        return "CREATE VIEW {} AS\n{}".format(
            self.escape_identifier(view), rows[0][0]
        )

    async def async_indexes_for_table(self, table):
        schema = self.schema or "public"
        rows = await self._async_fetch(
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
        )
        return [{"name": r[0], "unique": r[1], "sql": r[2]} for r in rows]

    async def async_label_column_details(self, table):
        schema = self.schema or "public"
        col_rows = await self._async_fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %(schema)s
              AND table_name = %(table)s
            ORDER BY ordinal_position
            """,
            {"schema": schema, "table": table},
        )
        unique_cols = set()
        idx_rows = await self._async_fetch(
            """
            SELECT a.attname
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            WHERE n.nspname = %(schema)s
              AND t.relname = %(table)s
              AND ix.indisunique = true
              AND ix.indnatts = 1
            """,
            {"schema": schema, "table": table},
        )
        for r in idx_rows:
            unique_cols.add(r[0])
        _text_types = {
            "text", "character varying", "varchar", "char", "character",
            "name", "citext",
        }
        details = {}
        for col_name, data_type in col_rows:
            py_type = str if data_type in _text_types else type(None)
            details[col_name] = (py_type, col_name in unique_cols)
        return details

    async def async_schema_version(self):
        schema = self.schema or "public"
        async with self._async_conn() as conn:
            cursor = await conn.execute(
                """
                SELECT md5(string_agg(
                    table_name || '.' || column_name || '.' || data_type,
                    ',' ORDER BY table_name, ordinal_position
                )) AS hash
                FROM information_schema.columns
                WHERE table_schema = %(schema)s
                """,
                {"schema": schema},
            )
            row = await cursor.fetchone()
        if row and row[0]:
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

    # ---- Write operations ----

    def table_schema_string(self, conn, table_name):
        return self.get_table_definition(conn, table_name)

    def _pg_type_for_value(self, value):
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "DOUBLE PRECISION"
        if isinstance(value, bytes):
            return "BYTEA"
        return "TEXT"

    def _ensure_columns(self, conn, table_name, rows, pk=None, alter=False):
        """Create table if needed, or alter to add missing columns."""
        escape = self.escape_identifier
        table_exists = self.table_exists(conn, table_name)

        if not table_exists:
            # Infer columns from rows
            all_columns = {}
            for row in rows:
                for col, val in row.items():
                    if col not in all_columns:
                        all_columns[col] = self._pg_type_for_value(val)
            pk_list = [pk] if isinstance(pk, str) else (pk or [])
            # Add auto-increment PK columns not present in rows
            for pk_col in pk_list:
                if pk_col not in all_columns:
                    all_columns[pk_col] = "SERIAL"
            # Build column defs in order: PK columns first, then data columns
            col_defs = []
            for col_name in pk_list:
                if col_name in all_columns:
                    col_defs.append(f"{escape(col_name)} {all_columns[col_name]}")
            for col_name, col_type in all_columns.items():
                if col_name not in pk_list:
                    col_defs.append(f"{escape(col_name)} {col_type}")
            if pk_list:
                pk_str = ", ".join(escape(c) for c in pk_list)
                col_defs.append(f"PRIMARY KEY ({pk_str})")
            sql = "CREATE TABLE {} ({})".format(
                escape(table_name), ", ".join(col_defs)
            )
            conn.execute(sql)
        elif alter:
            existing_cols = set(self.table_columns(conn, table_name))
            for row in rows:
                for col, val in row.items():
                    if col not in existing_cols:
                        col_type = self._pg_type_for_value(val)
                        conn.execute(
                            "ALTER TABLE {} ADD COLUMN {} {}".format(
                                escape(table_name), escape(col), col_type
                            )
                        )
                        existing_cols.add(col)

    def write_insert_rows(
        self,
        conn,
        table_name,
        rows,
        pk=None,
        alter=False,
        ignore=False,
        replace=False,
        return_rows=False,
    ):
        if not rows:
            return [] if return_rows else None

        escape = self.escape_identifier
        self._ensure_columns(conn, table_name, rows, pk=pk, alter=alter)

        pk_list = [pk] if isinstance(pk, str) else (pk or [])

        # Collect all column names across all rows
        all_cols = list(dict.fromkeys(col for row in rows for col in row))
        col_str = ", ".join(escape(c) for c in all_cols)

        # Build conflict clause
        conflict_clause = ""
        if pk_list and ignore:
            conflict_clause = " ON CONFLICT DO NOTHING"
        elif pk_list and replace:
            non_pk_cols = [c for c in all_cols if c not in pk_list]
            if non_pk_cols:
                update_parts = ", ".join(
                    "{} = EXCLUDED.{}".format(escape(c), escape(c))
                    for c in non_pk_cols
                )
                pk_str = ", ".join(escape(c) for c in pk_list)
                conflict_clause = (
                    " ON CONFLICT ({}) DO UPDATE SET {}".format(pk_str, update_parts)
                )
            else:
                conflict_clause = " ON CONFLICT DO NOTHING"

        returning = " RETURNING *" if return_rows else ""
        all_returned = []

        for row in rows:
            placeholders = ", ".join("%s" for _ in all_cols)
            values = [row.get(c) for c in all_cols]
            sql = "INSERT INTO {} ({}) VALUES ({}){}{}".format(
                escape(table_name), col_str, placeholders,
                conflict_clause, returning,
            )
            cursor = conn.execute(sql, values)
            if return_rows:
                result_rows = cursor.fetchall()
                for r in result_rows:
                    all_returned.append(dict(zip(r.keys(), r)))

        return all_returned if return_rows else None

    def write_upsert_rows(self, conn, table_name, rows, pk=None, alter=False):
        if not rows:
            return

        escape = self.escape_identifier
        self._ensure_columns(conn, table_name, rows, pk=pk, alter=alter)

        pk_list = [pk] if isinstance(pk, str) else (pk or [])

        for row in rows:
            cols = list(row.keys())
            col_str = ", ".join(escape(c) for c in cols)
            placeholders = ", ".join("%s" for _ in cols)
            values = list(row.values())

            non_pk_cols = [c for c in cols if c not in pk_list]
            if non_pk_cols:
                update_parts = ", ".join(
                    "{} = EXCLUDED.{}".format(escape(c), escape(c))
                    for c in non_pk_cols
                )
                pk_str = ", ".join(escape(c) for c in pk_list)
                conflict_clause = (
                    " ON CONFLICT ({}) DO UPDATE SET {}".format(pk_str, update_parts)
                )
            else:
                pk_str = ", ".join(escape(c) for c in pk_list)
                conflict_clause = " ON CONFLICT ({}) DO NOTHING".format(pk_str)

            sql = "INSERT INTO {} ({}) VALUES ({}){}".format(
                escape(table_name), col_str, placeholders, conflict_clause,
            )
            conn.execute(sql, values)

    def write_delete_row(self, conn, table_name, pks, pk_values):
        escape = self.escape_identifier
        where_parts = ["{} = %s".format(escape(pk)) for pk in pks]
        sql = "DELETE FROM {} WHERE {}".format(
            escape(table_name), " AND ".join(where_parts)
        )
        conn.execute(sql, list(pk_values))

    def write_update_row(
        self, conn, table_name, pks, pk_values, updates, alter=False
    ):
        escape = self.escape_identifier

        if alter:
            existing_cols = set(self.table_columns(conn, table_name))
            for col, val in updates.items():
                if col not in existing_cols:
                    col_type = self._pg_type_for_value(val)
                    conn.execute(
                        "ALTER TABLE {} ADD COLUMN {} {}".format(
                            escape(table_name), escape(col), col_type
                        )
                    )
                    existing_cols.add(col)

        set_parts = ["{} = %s".format(escape(col)) for col in updates]
        where_parts = ["{} = %s".format(escape(pk)) for pk in pks]
        sql = "UPDATE {} SET {} WHERE {}".format(
            escape(table_name),
            ", ".join(set_parts),
            " AND ".join(where_parts),
        )
        params = list(updates.values()) + list(pk_values)
        conn.execute(sql, params)

    def write_drop_table(self, conn, table_name):
        conn.execute("DROP TABLE {}".format(self.escape_identifier(table_name)))

    def write_create_table(self, conn, table_name, columns, pk=None):
        escape = self.escape_identifier
        _type_map = {
            "text": "TEXT",
            "integer": "INTEGER",
            "float": "DOUBLE PRECISION",
            "blob": "BYTEA",
        }
        pk_list = [pk] if isinstance(pk, str) else (pk or [])
        col_defs = []
        for col_name, col_type in columns.items():
            pg_type = _type_map.get(col_type, col_type.upper())
            col_defs.append("{} {}".format(escape(col_name), pg_type))
        if pk_list:
            pk_str = ", ".join(escape(c) for c in pk_list)
            col_defs.append("PRIMARY KEY ({})".format(pk_str))
        sql = "CREATE TABLE {} ({})".format(
            escape(table_name), ", ".join(col_defs)
        )
        conn.execute(sql)
        return self.get_table_definition(conn, table_name)


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
