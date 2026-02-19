"""SQLite database backend for Datasette.

This extracts all SQLite-specific logic from the old Database class:
- Connection creation with URI modes
- sqlite_master / PRAGMA introspection
- FTS detection
- sqlite_timelimit for query timeouts
- Single-writer thread queue
- escape_sqlite identifier quoting
"""

import asyncio
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import janus
import queue
import sys
import threading
import time
import uuid

from .base import DatabaseBackend, Column
from ..tracer import trace
from ..utils.sqlite import sqlite3, sqlite_version, supports_table_xinfo


connections = threading.local()

# From https://www.sqlite.org/lang_keywords.html
_reserved_words = set(
    (
        "abort action add after all alter analyze and as asc attach autoincrement "
        "before begin between by cascade case cast check collate column commit "
        "conflict constraint create cross current_date current_time "
        "current_timestamp database default deferrable deferred delete desc detach "
        "distinct drop each else end escape except exclusive exists explain fail "
        "for foreign from full glob group having if ignore immediate in index "
        "indexed initially inner insert instead intersect into is isnull join key "
        "left like limit match natural no not notnull null of offset on or order "
        "outer plan pragma primary query raise recursive references regexp reindex "
        "release rename replace restrict right rollback row savepoint select set "
        "table temp temporary then to transaction trigger union unique update using "
        "vacuum values view virtual when where with without"
    ).split()
)
import re

_boring_keyword_re = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class WriteTask:
    __slots__ = ("fn", "task_id", "reply_queue", "isolated_connection", "transaction")

    def __init__(self, fn, task_id, reply_queue, isolated_connection, transaction):
        self.fn = fn
        self.task_id = task_id
        self.reply_queue = reply_queue
        self.isolated_connection = isolated_connection
        self.transaction = transaction


class SQLiteBackend(DatabaseBackend):
    backend_type = "sqlite"
    _thread_local_id_counter = 1

    def __init__(
        self,
        ds=None,
        path=None,
        is_mutable=True,
        is_memory=False,
        memory_name=None,
        mode=None,
        nolock=False,
    ):
        self.ds = ds
        self.path = path
        self.is_mutable = is_mutable
        self.is_memory = is_memory
        self.memory_name = memory_name
        if memory_name is not None:
            self.is_memory = True
        self.mode = mode
        self.nolock = nolock

        self._thread_local_id = f"x{SQLiteBackend._thread_local_id_counter}"
        SQLiteBackend._thread_local_id_counter += 1

        # Connection tracking
        self._read_connection = None
        self._write_connection = None
        self._all_file_connections = []

        # Write thread (SQLite-specific: single writer)
        self._write_thread = None
        self._write_queue = None

    # ---- Connection lifecycle ----

    def create_connection(self, write=False):
        extra_kwargs = {}
        if write:
            extra_kwargs["isolation_level"] = "IMMEDIATE"
        if self.memory_name:
            uri = "file:{}?mode=memory&cache=shared".format(self.memory_name)
            conn = sqlite3.connect(
                uri, uri=True, check_same_thread=False, **extra_kwargs
            )
            if not write:
                conn.execute("PRAGMA query_only=1")
            return conn
        if self.is_memory:
            return sqlite3.connect(":memory:", uri=True)

        if self.is_mutable:
            qs = "?mode=ro"
            if self.nolock:
                qs += "&nolock=1"
        else:
            qs = "?immutable=1"
        assert not (write and not self.is_mutable)
        if write:
            qs = ""
        if self.mode is not None:
            qs = f"?mode={self.mode}"
        conn = sqlite3.connect(
            f"file:{self.path}{qs}", uri=True, check_same_thread=False, **extra_kwargs
        )
        self._all_file_connections.append(conn)
        return conn

    def close_connection(self, conn):
        conn.close()

    def close_all(self):
        for connection in self._all_file_connections:
            connection.close()

    def prepare_connection(self, conn, datasette, database_name):
        conn.row_factory = sqlite3.Row
        conn.text_factory = lambda x: str(x, "utf-8", "replace")
        if datasette and hasattr(datasette, "sqlite_extensions"):
            if datasette.sqlite_extensions and database_name != "__INTERNAL__":
                conn.enable_load_extension(True)
                for extension in datasette.sqlite_extensions:
                    if isinstance(extension, tuple):
                        path, entrypoint = extension
                        conn.execute(
                            "SELECT load_extension(?, ?)", [path, entrypoint]
                        )
                    else:
                        conn.execute("SELECT load_extension(?)", [extension])
        if datasette and datasette.setting("cache_size_kb"):
            conn.execute(
                f"PRAGMA cache_size=-{datasette.setting('cache_size_kb')}"
            )

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

        def sql_operation_in_thread(conn):
            time_limit_ms = self.ds.sql_time_limit_ms if self.ds else 1000
            if custom_time_limit and custom_time_limit < time_limit_ms:
                time_limit_ms = custom_time_limit

            with self.time_limit_context(conn, time_limit_ms):
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql, params if params is not None else {})
                    max_returned_rows = self.ds.max_returned_rows if self.ds else 100
                    if max_returned_rows == page_size:
                        max_returned_rows += 1
                    if max_returned_rows and truncate:
                        rows = cursor.fetchmany(max_returned_rows + 1)
                        truncated = len(rows) > max_returned_rows
                        rows = rows[:max_returned_rows]
                    else:
                        rows = cursor.fetchall()
                        truncated = False
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    if e.args == ("interrupted",):
                        from ..database import QueryInterrupted

                        raise QueryInterrupted(e, sql, params)
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

        with trace("sql", database=getattr(self, "_database_name", ""), sql=sql.strip(), params=params):
            results = await self.execute_fn(sql_operation_in_thread)
        return results

    def _do_prepare_connection(self, conn):
        """Prepare a connection using Datasette._prepare_connection if available,
        falling back to self.prepare_connection for standalone usage."""
        name = getattr(self, "_database_name", "")
        if self.ds and hasattr(self.ds, "_prepare_connection"):
            self.ds._prepare_connection(conn, name)
        elif self.ds:
            self.prepare_connection(conn, self.ds, name)

    async def execute_fn(self, fn):
        if self.ds is None or self.ds.executor is None:
            # non-threaded mode
            if self._read_connection is None:
                self._read_connection = self.create_connection()
                if self.ds:
                    self._do_prepare_connection(self._read_connection)
            return fn(self._read_connection)

        # threaded mode
        def in_thread():
            conn = getattr(connections, self._thread_local_id, None)
            if not conn:
                conn = self.create_connection()
                self._do_prepare_connection(conn)
                setattr(connections, self._thread_local_id, conn)
            return fn(conn)

        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )

    async def execute_write(self, sql, params=None, block=True, request=None):
        def _inner(conn):
            return conn.execute(sql, params or [])

        with trace("sql", database=getattr(self, "_database_name", ""), sql=sql.strip(), params=params):
            results = await self.execute_write_fn(_inner, block=block, request=request)
        return results

    async def execute_write_script(self, sql, block=True, request=None):
        def _inner(conn):
            return conn.executescript(sql)

        with trace("sql", database=getattr(self, "_database_name", ""), sql=sql.strip(), executescript=True):
            results = await self.execute_write_fn(
                _inner, block=block, transaction=False, request=request
            )
        return results

    async def execute_write_many(self, sql, params_seq, block=True, request=None):
        def _inner(conn):
            count = 0

            def count_params(params):
                nonlocal count
                for param in params:
                    count += 1
                    yield param

            return conn.executemany(sql, count_params(params_seq)), count

        with trace(
            "sql", database=getattr(self, "_database_name", ""), sql=sql.strip(), executemany=True
        ) as kwargs:
            results, count = await self.execute_write_fn(
                _inner, block=block, request=request
            )
            kwargs["count"] = count
        return results

    async def execute_write_fn(self, fn, block=True, transaction=True, request=None):
        fn = self._wrap_fn_with_hooks(fn, request, transaction)
        if self.ds is None or self.ds.executor is None:
            # non-threaded mode
            if self._write_connection is None:
                self._write_connection = self.create_connection(write=True)
                if self.ds:
                    self._do_prepare_connection(self._write_connection)
            if transaction:
                with self._write_connection:
                    return fn(self._write_connection)
            else:
                return fn(self._write_connection)
        else:
            return await self._send_to_write_thread(
                fn, block=block, transaction=transaction
            )

    async def execute_isolated_fn(self, fn):
        if self.ds is None or self.ds.executor is None:
            isolated_connection = self.create_connection(write=True)
            try:
                result = fn(isolated_connection)
            finally:
                isolated_connection.close()
                try:
                    self._all_file_connections.remove(isolated_connection)
                except ValueError:
                    pass
            return result
        else:
            return await self._send_to_write_thread(fn, isolated_connection=True)

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

    async def _send_to_write_thread(
        self, fn, block=True, isolated_connection=False, transaction=True
    ):
        if self._write_queue is None:
            self._write_queue = queue.Queue()
        if self._write_thread is None:
            self._write_thread = threading.Thread(
                target=self._execute_writes, daemon=True
            )
            self._write_thread.name = "_execute_writes for database {}".format(
                getattr(self, "_database_name", "")
            )
            self._write_thread.start()
        task_id = uuid.uuid5(uuid.NAMESPACE_DNS, "datasette.io")
        reply_queue = janus.Queue()
        self._write_queue.put(
            WriteTask(fn, task_id, reply_queue, isolated_connection, transaction)
        )
        if block:
            result = await reply_queue.async_q.get()
            if isinstance(result, Exception):
                raise result
            else:
                return result
        else:
            return task_id

    def _execute_writes(self):
        conn_exception = None
        conn = None
        try:
            conn = self.create_connection(write=True)
            if self.ds:
                self._do_prepare_connection(conn)
        except Exception as e:
            conn_exception = e
        while True:
            task = self._write_queue.get()
            if conn_exception is not None:
                result = conn_exception
            else:
                if task.isolated_connection:
                    isolated_connection = self.create_connection(write=True)
                    try:
                        result = task.fn(isolated_connection)
                    except Exception as e:
                        sys.stderr.write("{}\n".format(e))
                        sys.stderr.flush()
                        result = e
                    finally:
                        isolated_connection.close()
                        try:
                            self._all_file_connections.remove(isolated_connection)
                        except ValueError:
                            pass
                else:
                    try:
                        if task.transaction:
                            with conn:
                                result = task.fn(conn)
                        else:
                            result = task.fn(conn)
                    except Exception as e:
                        sys.stderr.write("{}\n".format(e))
                        sys.stderr.flush()
                        result = e
            task.reply_queue.sync_q.put(result)

    # ---- SQL dialect ----

    def translate_sql(self, sql):
        # SQLite uses :name natively - no translation needed
        return sql

    def escape_identifier(self, identifier):
        if _boring_keyword_re.match(identifier) and (
            identifier.lower() not in _reserved_words
        ):
            return identifier
        else:
            return f"[{identifier}]"

    # ---- Time limiting ----

    @contextmanager
    def time_limit_context(self, conn, ms):
        deadline = time.perf_counter() + (ms / 1000)
        n = 1000
        if ms <= 20:
            n = 1

        def handler():
            if time.perf_counter() >= deadline:
                return 1

        conn.set_progress_handler(handler, n)
        try:
            yield
        finally:
            conn.set_progress_handler(None, n)

    def is_interrupted_error(self, error):
        return (
            isinstance(error, sqlite3.OperationalError)
            and error.args == ("interrupted",)
        )

    def is_operational_error(self, error):
        return isinstance(error, (sqlite3.OperationalError, sqlite3.DatabaseError))

    # ---- Schema introspection ----

    def table_names(self, conn):
        rows = conn.execute(
            "select name from sqlite_master where type='table' order by name"
        ).fetchall()
        return [r[0] for r in rows]

    def view_names(self, conn):
        rows = conn.execute(
            "select name from sqlite_master where type='view'"
        ).fetchall()
        return [r[0] for r in rows]

    def table_exists(self, conn, table):
        rows = conn.execute(
            "select 1 from sqlite_master where type='table' and name=?",
            (table,),
        ).fetchall()
        return bool(rows)

    def view_exists(self, conn, view):
        rows = conn.execute(
            "select 1 from sqlite_master where type='view' and name=?",
            (view,),
        ).fetchall()
        return bool(rows)

    def table_columns(self, conn, table):
        return [col.name for col in self.table_column_details(conn, table)]

    def table_column_details(self, conn, table):
        if supports_table_xinfo():
            return [
                Column(*r)
                for r in conn.execute(
                    f"PRAGMA table_xinfo({self.escape_identifier(table)});"
                ).fetchall()
            ]
        else:
            conn.execute("select 1 from sqlite_master limit 1").fetchall()
            return [
                Column(*(list(r) + [0]))
                for r in conn.execute(
                    f"PRAGMA table_info({self.escape_identifier(table)});"
                ).fetchall()
            ]

    def primary_keys(self, conn, table):
        columns = self.table_column_details(conn, table)
        pks = [column for column in columns if column.is_pk]
        pks.sort(key=lambda column: column.is_pk)
        return [column.name for column in pks]

    def foreign_keys_for_table(self, conn, table):
        infos = conn.execute(
            f"PRAGMA foreign_key_list([{table}])"
        ).fetchall()
        fks = []
        for info in infos:
            if info is not None:
                id, seq, table_name, from_, to_, on_update, on_delete, match = info
                fks.append(
                    {
                        "column": from_,
                        "other_table": table_name,
                        "other_column": to_,
                        "id": id,
                        "seq": seq,
                    }
                )
        id_counts = Counter(fk["id"] for fk in fks)
        return [
            {
                "column": fk["column"],
                "other_table": fk["other_table"],
                "other_column": fk["other_column"],
            }
            for fk in fks
            if id_counts[fk["id"]] == 1
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
        hidden_tables = []
        if sqlite_version()[1] >= 37:
            hidden_tables += [
                x[0]
                for x in conn.execute("""
                    with shadow_tables as (
                        select name
                        from pragma_table_list
                        where [type] = 'shadow'
                        order by name
                    ),
                    core_tables as (
                        select name
                        from sqlite_master
                        WHERE name in ('sqlite_stat1', 'sqlite_stat2', 'sqlite_stat3', 'sqlite_stat4')
                          OR substr(name, 1, 1) == '_'
                    ),
                    combined as (
                        select name from shadow_tables
                        union all
                        select name from core_tables
                    )
                    select name from combined order by 1
                """).fetchall()
            ]
        else:
            hidden_tables += [
                x[0]
                for x in conn.execute("""
                    WITH base AS (
                        SELECT name
                        FROM sqlite_master
                        WHERE name IN ('sqlite_stat1', 'sqlite_stat2', 'sqlite_stat3', 'sqlite_stat4')
                          OR substr(name, 1, 1) == '_'
                    ),
                    fts_suffixes AS (
                        SELECT column1 AS suffix
                        FROM (VALUES ('_data'), ('_idx'), ('_docsize'), ('_content'), ('_config'))
                    ),
                    fts5_names AS (
                        SELECT name
                        FROM sqlite_master
                        WHERE sql LIKE '%VIRTUAL TABLE%USING FTS%'
                    ),
                    fts5_shadow_tables AS (
                        SELECT
                            printf('%s%s', fts5_names.name, fts_suffixes.suffix) AS name
                        FROM fts5_names
                        JOIN fts_suffixes
                    ),
                    fts3_suffixes AS (
                        SELECT column1 AS suffix
                        FROM (VALUES ('_content'), ('_segdir'), ('_segments'), ('_stat'), ('_docsize'))
                    ),
                    fts3_names AS (
                        SELECT name
                        FROM sqlite_master
                        WHERE sql LIKE '%VIRTUAL TABLE%USING FTS3%'
                          OR sql LIKE '%VIRTUAL TABLE%USING FTS4%'
                    ),
                    fts3_shadow_tables AS (
                        SELECT
                            printf('%s%s', fts3_names.name, fts3_suffixes.suffix) AS name
                        FROM fts3_names
                        JOIN fts3_suffixes
                    ),
                    final AS (
                        SELECT name FROM base
                        UNION ALL
                        SELECT name FROM fts5_shadow_tables
                        UNION ALL
                        SELECT name FROM fts3_shadow_tables
                    )
                    SELECT name FROM final ORDER BY 1
                """).fetchall()
            ]

        # Also hide FTS tables with content= argument
        hidden_tables += [
            x[0]
            for x in conn.execute("""
                SELECT name
                FROM sqlite_master
                WHERE sql LIKE '%VIRTUAL TABLE%'
                  AND sql LIKE '%USING FTS%'
                  AND sql LIKE '%content=%'
            """).fetchall()
        ]

        # Spatialite tables
        if self._detect_spatialite(conn):
            hidden_tables += [
                "ElementaryGeometries",
                "SpatialIndex",
                "geometry_columns",
                "spatial_ref_sys",
                "spatialite_history",
                "sql_statements_log",
                "sqlite_sequence",
                "views_geometry_columns",
                "virts_geometry_columns",
                "data_licenses",
                "KNN",
                "KNN2",
            ] + [
                r[0]
                for r in conn.execute(
                    """
                    select name from sqlite_master
                    where name like "idx_%"
                    and type = "table"
                """
                ).fetchall()
            ]

        return hidden_tables

    def _detect_spatialite(self, conn):
        rows = conn.execute(
            'select 1 from sqlite_master where tbl_name = "geometry_columns"'
        ).fetchall()
        return len(rows) > 0

    def get_table_definition(self, conn, table, type_="table"):
        table_definition_rows = conn.execute(
            "select sql from sqlite_master where name = :n and type=:t",
            {"n": table, "t": type_},
        ).fetchall()
        if not table_definition_rows:
            return None
        bits = [table_definition_rows[0][0] + ";"]
        index_rows = conn.execute(
            "select sql from sqlite_master where tbl_name = :n and type='index' and sql is not null",
            {"n": table},
        ).fetchall()
        for index_row in index_rows:
            bits.append(index_row[0] + ";")
        return "\n".join(bits)

    def get_view_definition(self, conn, view):
        return self.get_table_definition(conn, view, "view")

    def indexes_for_table(self, conn, table):
        rows = conn.execute(f"PRAGMA index_list([{table}])").fetchall()
        return [dict(r) for r in rows]

    def label_column_details(self, conn, table):
        import sqlite_utils

        db = sqlite_utils.Database(conn)
        columns = db[table].columns_dict
        indexes = db[table].indexes
        details = {}
        for name in columns:
            is_unique = any(
                index
                for index in indexes
                if index.columns == [name] and index.unique
            )
            details[name] = (columns[name], is_unique)
        return details

    def detect_fts(self, conn, table):
        sql = r"""
            select name from sqlite_master
                where rootpage = 0
                and (
                    sql like '%VIRTUAL TABLE%USING FTS%content="{table}"%'
                    or sql like '%VIRTUAL TABLE%USING FTS%content=[{table}]%'
                    or (
                        tbl_name = "{table}"
                        and sql like '%VIRTUAL TABLE%USING FTS%'
                    )
                )
        """.format(table=table.replace("'", "''"))
        rows = conn.execute(sql).fetchall()
        if len(rows) == 0:
            return None
        else:
            return rows[0][0]

    def supports_fts(self):
        return True

    def schema_version(self, conn):
        return conn.execute("PRAGMA schema_version").fetchone()[0]

    def suggest_name(self):
        if self.path:
            return Path(self.path).stem
        elif self.memory_name:
            return self.memory_name
        else:
            return "db"

    # ---- Write operations ----

    def table_schema_string(self, conn, table_name):
        import sqlite_utils

        return sqlite_utils.Database(conn)[table_name].schema

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
        import sqlite_utils

        table = sqlite_utils.Database(conn)[table_name]
        kwargs = {"ignore": ignore, "replace": replace, "alter": alter}
        if pk is not None:
            kwargs["pk"] = pk
        if return_rows:
            rowids = []
            for row in rows:
                rowids.append(table.insert(row, **kwargs).last_rowid)
            return list(
                table.rows_where(
                    "rowid in ({})".format(",".join("?" for _ in rowids)),
                    rowids,
                )
            )
        else:
            table.insert_all(rows, **kwargs)
            return None

    def write_upsert_rows(self, conn, table_name, rows, pk=None, alter=False):
        import sqlite_utils

        table = sqlite_utils.Database(conn)[table_name]
        table.upsert_all(rows, pk=pk, alter=alter)

    def write_delete_row(self, conn, table_name, pks, pk_values):
        import sqlite_utils

        pk_val = pk_values[0] if len(pk_values) == 1 else tuple(pk_values)
        sqlite_utils.Database(conn)[table_name].delete(pk_val)

    def write_update_row(self, conn, table_name, pks, pk_values, updates, alter=False):
        import sqlite_utils

        pk_val = pk_values[0] if len(pk_values) == 1 else tuple(pk_values)
        sqlite_utils.Database(conn)[table_name].update(pk_val, updates, alter=alter)

    def write_drop_table(self, conn, table_name):
        import sqlite_utils

        sqlite_utils.Database(conn)[table_name].drop()

    def write_create_table(self, conn, table_name, columns, pk=None):
        import sqlite_utils

        db = sqlite_utils.Database(conn)
        db[table_name].create(columns, pk=pk)
        return db[table_name].schema


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
