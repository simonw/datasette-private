from collections import namedtuple
from pathlib import Path
import sqlite_utils
import sys

from .utils import (
    detect_fts,
    detect_primary_keys,
    detect_spatialite,
    get_all_foreign_keys,
    get_outbound_foreign_keys,
    md5_not_usedforsecurity,
    sqlite_timelimit,
    sqlite3,
    table_columns,
    table_column_details,
)
from .utils.sqlite import sqlite_version
from .inspect import inspect_hash
from .tracer import trace

AttachedDatabase = namedtuple("AttachedDatabase", ("seq", "name", "file"))


class Database:
    # For table counts stop at this many rows:
    count_limit = 10000

    def __init__(
        self,
        ds,
        path=None,
        is_mutable=True,
        is_memory=False,
        memory_name=None,
        mode=None,
        backend=None,
    ):
        self.name = None
        self.route = None
        self.ds = ds
        self.cached_hash = None
        self.cached_size = None
        self._cached_table_counts = None

        # Create backend - SQLiteBackend by default for backward compatibility
        if backend is not None:
            self.backend = backend
        else:
            from .backends.sqlite import SQLiteBackend

            self.backend = SQLiteBackend(
                ds=ds,
                path=path,
                is_mutable=is_mutable,
                is_memory=is_memory,
                memory_name=memory_name,
                mode=mode,
                nolock=getattr(ds, "nolock", False),
            )

        # Expose backend properties for backward compatibility
        self.path = getattr(self.backend, "path", path)
        self.is_mutable = getattr(self.backend, "is_mutable", is_mutable)
        self.is_memory = getattr(self.backend, "is_memory", is_memory)
        self.memory_name = getattr(self.backend, "memory_name", memory_name)
        self.mode = getattr(self.backend, "mode", mode)

    def _set_name(self, name):
        """Called by Datasette.add_database to set the database name."""
        self.name = name
        # Tell the backend its database name for tracing etc.
        self.backend._database_name = name

    @property
    def cached_table_counts(self):
        if self._cached_table_counts is not None:
            return self._cached_table_counts
        if self.ds.inspect_data and self.ds.inspect_data.get(self.name):
            self._cached_table_counts = {
                key: value["count"]
                for key, value in self.ds.inspect_data[self.name]["tables"].items()
            }
        return self._cached_table_counts

    @property
    def color(self):
        if self.hash:
            return self.hash[:6]
        return md5_not_usedforsecurity(self.name)[:6]

    def suggest_name(self):
        return self.backend.suggest_name()

    # ---- Connection management (delegated) ----

    def connect(self, write=False):
        return self.backend.create_connection(write=write)

    def close(self):
        if hasattr(self.backend, "close_all"):
            self.backend.close_all()
        elif hasattr(self.backend, "_all_file_connections"):
            for conn in self.backend._all_file_connections:
                conn.close()

    # ---- Execution (delegated to backend) ----

    async def execute(
        self,
        sql,
        params=None,
        truncate=False,
        custom_time_limit=None,
        page_size=None,
        log_sql_errors=True,
    ):
        """Executes sql against this database."""
        return await self.backend.execute(
            sql,
            params=params,
            truncate=truncate,
            custom_time_limit=custom_time_limit,
            page_size=page_size,
            log_sql_errors=log_sql_errors,
        )

    async def execute_fn(self, fn):
        return await self.backend.execute_fn(fn)

    async def execute_write(self, sql, params=None, block=True, request=None):
        with trace("sql", database=self.name, sql=sql.strip(), params=params):
            return await self.backend.execute_write(
                sql, params=params, block=block, request=request
            )

    async def execute_write_script(self, sql, block=True, request=None):
        with trace("sql", database=self.name, sql=sql.strip(), executescript=True):
            return await self.backend.execute_write_script(
                sql, block=block, request=request
            )

    async def execute_write_many(self, sql, params_seq, block=True, request=None):
        with trace(
            "sql", database=self.name, sql=sql.strip(), executemany=True
        ) as kwargs:
            result = await self.backend.execute_write_many(
                sql, params_seq, block=block, request=request
            )
            if isinstance(result, tuple):
                kwargs["count"] = result[1]
                return result[0]
            return result

    async def execute_write_fn(self, fn, block=True, transaction=True, request=None):
        return await self.backend.execute_write_fn(
            fn, block=block, transaction=transaction, request=request
        )

    async def execute_isolated_fn(self, fn):
        return await self.backend.execute_isolated_fn(fn)

    # ---- SQL dialect ----

    def escape_identifier(self, identifier):
        return self.backend.escape_identifier(identifier)

    # ---- Schema introspection (delegated to backend via execute_fn) ----

    async def table_names(self):
        return await self.execute_fn(lambda conn: self.backend.table_names(conn))

    async def view_names(self):
        return await self.execute_fn(lambda conn: self.backend.view_names(conn))

    async def table_exists(self, table):
        return await self.execute_fn(lambda conn: self.backend.table_exists(conn, table))

    async def view_exists(self, table):
        return await self.execute_fn(lambda conn: self.backend.view_exists(conn, table))

    async def table_columns(self, table):
        return await self.execute_fn(lambda conn: self.backend.table_columns(conn, table))

    async def table_column_details(self, table):
        return await self.execute_fn(lambda conn: self.backend.table_column_details(conn, table))

    async def primary_keys(self, table):
        return await self.execute_fn(lambda conn: self.backend.primary_keys(conn, table))

    async def fts_table(self, table):
        return await self.execute_fn(lambda conn: self.backend.detect_fts(conn, table))

    async def foreign_keys_for_table(self, table):
        return await self.execute_fn(
            lambda conn: self.backend.foreign_keys_for_table(conn, table)
        )

    async def get_all_foreign_keys(self):
        return await self.execute_fn(lambda conn: self.backend.get_all_foreign_keys(conn))

    async def hidden_table_names(self):
        hidden_tables = []
        # Add any tables marked as hidden in config
        db_config = self.ds.config.get("databases", {}).get(self.name, {})
        if "tables" in db_config:
            hidden_tables += [
                t
                for t in db_config["tables"]
                if db_config["tables"][t].get("hidden")
            ]
        # Get backend-specific hidden tables
        hidden_tables += await self.execute_fn(
            lambda conn: self.backend.hidden_table_names(conn)
        )
        return hidden_tables

    async def get_table_definition(self, table, type_="table"):
        return await self.execute_fn(
            lambda conn: self.backend.get_table_definition(conn, table)
            if type_ == "table"
            else self.backend.get_view_definition(conn, table)
        )

    async def get_view_definition(self, view):
        return await self.execute_fn(
            lambda conn: self.backend.get_view_definition(conn, view)
        )

    async def label_column_for_table(self, table):
        explicit_label_column = (await self.ds.table_config(self.name, table)).get(
            "label_column"
        )
        if explicit_label_column:
            return explicit_label_column

        def column_details(conn):
            return self.backend.label_column_details(conn, table)

        column_details = await self.execute_fn(column_details)
        unique_text_columns = [
            name
            for name, (type_, is_unique) in column_details.items()
            if is_unique and type_ is str
        ]
        if len(unique_text_columns) == 1:
            return unique_text_columns[0]

        column_names = list(column_details.keys())
        name_or_title = [c for c in column_names if c.lower() in ("name", "title")]
        if name_or_title:
            return name_or_title[0]
        if (
            column_names
            and len(column_names) == 2
            and ("id" in column_names or "pk" in column_names)
            and not set(column_names) == {"id", "pk"}
        ):
            return [c for c in column_names if c not in ("id", "pk")][0]
        return None

    # ---- Properties ----

    @property
    def hash(self):
        if self.cached_hash is not None:
            return self.cached_hash
        elif self.is_mutable or self.is_memory or not self.path:
            return None
        elif self.ds.inspect_data and self.ds.inspect_data.get(self.name):
            self.cached_hash = self.ds.inspect_data[self.name]["hash"]
            return self.cached_hash
        else:
            p = Path(self.path)
            self.cached_hash = inspect_hash(p)
            return self.cached_hash

    @property
    def size(self):
        if self.cached_size is not None:
            return self.cached_size
        elif self.is_memory or not self.path:
            return 0
        elif self.is_mutable:
            return Path(self.path).stat().st_size
        elif self.ds.inspect_data and self.ds.inspect_data.get(self.name):
            self.cached_size = self.ds.inspect_data[self.name]["size"]
            return self.cached_size
        else:
            self.cached_size = Path(self.path).stat().st_size
            return self.cached_size

    async def table_counts(self, limit=10):
        if not self.is_mutable and self.cached_table_counts is not None:
            return self.cached_table_counts
        counts = {}
        escape = self.backend.escape_identifier
        for table in await self.table_names():
            try:
                table_count = (
                    await self.execute(
                        f"select count(*) from (select * from {escape(table)} limit {self.count_limit + 1})",
                        custom_time_limit=limit,
                    )
                ).rows[0][0]
                counts[table] = table_count
            except Exception:
                counts[table] = None
        if not self.is_mutable:
            self._cached_table_counts = counts
        return counts

    @property
    def mtime_ns(self):
        if self.is_memory or not self.path:
            return None
        return Path(self.path).stat().st_mtime_ns

    async def attached_databases(self):
        if self.backend.backend_type != "sqlite":
            return []
        results = await self.execute("PRAGMA database_list;")
        return [
            AttachedDatabase(*row)
            for row in results.rows
            if row["seq"] > 0 and row["name"] != "temp"
        ]

    def __repr__(self):
        tags = []
        if self.is_mutable:
            tags.append("mutable")
        if self.is_memory:
            tags.append("memory")
        if self.hash:
            tags.append(f"hash={self.hash}")
        if self.size is not None:
            tags.append(f"size={self.size}")
        tags_str = ""
        if tags:
            tags_str = f" ({', '.join(tags)})"
        return f"<Database: {self.name}{tags_str}>"


class QueryInterrupted(Exception):
    def __init__(self, e, sql, params):
        self.e = e
        self.sql = sql
        self.params = params

    def __str__(self):
        return "QueryInterrupted: {}".format(self.e)


class MultipleValues(Exception):
    pass


class Results:
    def __init__(self, rows, truncated, description):
        self.rows = rows
        self.truncated = truncated
        self.description = description

    @property
    def columns(self):
        return [d[0] for d in self.description]

    def first(self):
        if self.rows:
            return self.rows[0]
        else:
            return None

    def single_value(self):
        if self.rows and 1 == len(self.rows) and 1 == len(self.rows[0]):
            return self.rows[0][0]
        else:
            raise MultipleValues

    def dicts(self):
        return [dict(row) for row in self.rows]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)
