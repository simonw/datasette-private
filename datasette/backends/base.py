"""Abstract base class for database backends and shared types."""

from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


@runtime_checkable
class RowProtocol(Protocol):
    """The row interface that all backends must satisfy.

    sqlite3.Row already conforms to this protocol. Other backends
    must provide a row class that does too.
    """

    def __getitem__(self, key: int | str) -> Any: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def keys(self) -> Sequence[str]: ...


Column = namedtuple(
    "Column", ("cid", "name", "type", "notnull", "default_value", "is_pk", "hidden")
)


class DatabaseBackend(ABC):
    """Abstract interface for database backends.

    Each backend owns its concurrency strategy (e.g., SQLite uses a single-writer
    thread queue; PostgreSQL uses direct connections with no queue).
    """

    backend_type: str = ""

    # ---- Connection lifecycle ----

    @abstractmethod
    def create_connection(self, write: bool = False) -> Any:
        """Return a new database connection object."""
        ...

    def close_connection(self, conn: Any) -> None:
        """Close a single connection."""
        conn.close()

    @abstractmethod
    def prepare_connection(self, conn: Any, datasette: Any, database_name: str) -> None:
        """Configure a fresh connection (row factory, extensions, etc.)."""
        ...

    # ---- Async execution (backends own their concurrency) ----

    @abstractmethod
    async def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        truncate: bool = False,
        custom_time_limit: Optional[int] = None,
        page_size: Optional[int] = None,
        log_sql_errors: bool = True,
    ) -> Any:
        """Execute a read-only SQL query. Returns a Results object."""
        ...

    @abstractmethod
    async def execute_fn(self, fn) -> Any:
        """Run fn(conn) using the backend's read connection strategy."""
        ...

    @abstractmethod
    async def execute_write_fn(
        self, fn, block: bool = True, transaction: bool = True, request: Any = None
    ) -> Any:
        """Run fn(conn) using the backend's write connection strategy."""
        ...

    async def execute_write(self, sql: str, params=None, block: bool = True, request=None):
        """Execute a single write statement."""

        def _inner(conn):
            return conn.execute(sql, params or [])

        return await self.execute_write_fn(_inner, block=block, request=request)

    async def execute_write_script(self, sql: str, block: bool = True, request=None):
        """Execute multiple statements (like SQLite's executescript)."""

        def _inner(conn):
            return conn.executescript(sql)

        return await self.execute_write_fn(_inner, block=block, transaction=False, request=request)

    async def execute_write_many(self, sql: str, params_seq, block: bool = True, request=None):
        """Execute a statement with a sequence of parameter sets."""

        def _inner(conn):
            return conn.executemany(sql, params_seq)

        return await self.execute_write_fn(_inner, block=block, request=request)

    async def execute_isolated_fn(self, fn) -> Any:
        """Execute fn on a dedicated connection, blocking the write queue."""
        return await self.execute_write_fn(fn)

    # ---- SQL dialect ----

    def translate_sql(self, sql: str) -> str:
        """Translate SQL for this backend's dialect (e.g., param binding style).
        Default: no translation."""
        return sql

    @abstractmethod
    def escape_identifier(self, identifier: str) -> str:
        """Quote a table/column name for safe inclusion in SQL."""
        ...

    # ---- Time limiting ----

    @contextmanager
    def time_limit_context(self, conn: Any, ms: int):
        """Context manager that enforces a query time limit.
        Default: no-op (backends override with their mechanism)."""
        yield

    def is_interrupted_error(self, error: Exception) -> bool:
        """Return True if this exception represents a query timeout/interrupt."""
        return False

    def is_operational_error(self, error: Exception) -> bool:
        """Return True if this is an operational/database error."""
        return False

    # ---- Schema introspection (sync, called within execute_fn) ----

    @abstractmethod
    def table_names(self, conn: Any) -> List[str]:
        ...

    @abstractmethod
    def view_names(self, conn: Any) -> List[str]:
        ...

    def table_exists(self, conn: Any, table: str) -> bool:
        return table in self.table_names(conn)

    def view_exists(self, conn: Any, view: str) -> bool:
        return view in self.view_names(conn)

    @abstractmethod
    def table_columns(self, conn: Any, table: str) -> List[str]:
        ...

    @abstractmethod
    def table_column_details(self, conn: Any, table: str) -> List[Column]:
        ...

    @abstractmethod
    def primary_keys(self, conn: Any, table: str) -> List[str]:
        ...

    @abstractmethod
    def foreign_keys_for_table(self, conn: Any, table: str) -> List[Dict]:
        ...

    @abstractmethod
    def get_all_foreign_keys(self, conn: Any) -> Dict:
        ...

    def hidden_table_names(self, conn: Any) -> List[str]:
        """Return table names that should be hidden from the UI.
        Default: empty list."""
        return []

    def get_table_definition(self, conn: Any, table: str) -> Optional[str]:
        """Return the CREATE TABLE SQL or equivalent. Default: None."""
        return None

    def get_view_definition(self, conn: Any, view: str) -> Optional[str]:
        """Return the CREATE VIEW SQL or equivalent. Default: None."""
        return None

    def indexes_for_table(self, conn: Any, table: str) -> List[Dict]:
        """Return list of index dicts for a table. Default: empty."""
        return []

    def label_column_details(self, conn: Any, table: str) -> Dict:
        """Return {column_name: (python_type, is_unique)} for label column detection."""
        return {}

    def detect_fts(self, conn: Any, table: str) -> Optional[str]:
        """Return the name of the FTS table for this table, if any."""
        return None

    def supports_fts(self) -> bool:
        return False

    def schema_version(self, conn: Any) -> int:
        """Return a version number that changes when the schema changes.
        Used to determine if cached schema info needs refreshing."""
        return 0

    def suggest_name(self) -> str:
        """Suggest a name for this database based on connection info."""
        return "db"

    # ---- Write operations ----
    # These allow views to perform writes without depending on sqlite_utils.

    def table_schema_string(self, conn: Any, table_name: str) -> Optional[str]:
        """Return a CREATE TABLE statement string for a table."""
        return self.get_table_definition(conn, table_name)

    def write_insert_rows(
        self,
        conn: Any,
        table_name: str,
        rows: List[Dict],
        pk: Any = None,
        alter: bool = False,
        ignore: bool = False,
        replace: bool = False,
        return_rows: bool = False,
    ) -> Optional[List[Dict]]:
        """Insert rows into a table. Returns list of row dicts if return_rows."""
        raise NotImplementedError

    def write_upsert_rows(
        self,
        conn: Any,
        table_name: str,
        rows: List[Dict],
        pk: Any = None,
        alter: bool = False,
    ) -> None:
        """Upsert rows (insert or update on conflict)."""
        raise NotImplementedError

    def write_delete_row(
        self, conn: Any, table_name: str, pks: List[str], pk_values: List
    ) -> None:
        """Delete a row by primary key."""
        raise NotImplementedError

    def write_update_row(
        self,
        conn: Any,
        table_name: str,
        pks: List[str],
        pk_values: List,
        updates: Dict,
        alter: bool = False,
    ) -> None:
        """Update a row by primary key."""
        raise NotImplementedError

    def write_drop_table(self, conn: Any, table_name: str) -> None:
        """Drop a table."""
        raise NotImplementedError

    def write_create_table(
        self, conn: Any, table_name: str, columns: Dict[str, str], pk: Any = None
    ) -> str:
        """Create a table. columns is {name: type_str}. Returns schema string."""
        raise NotImplementedError
