"""Tests for the pluggable database backend system."""

import pytest
import pytest_asyncio
import sqlite3


@pytest.fixture
def sqlite_backend_with_tables():
    """Create a SQLiteBackend with test tables populated."""
    import secrets
    from datasette.backends.sqlite import SQLiteBackend

    name = f"test_introspection_{secrets.token_hex(4)}"
    backend = SQLiteBackend(is_memory=True, memory_name=name)
    conn = backend.create_connection(write=True)
    conn.executescript("""
        CREATE TABLE simple_primary_key (
            id integer primary key,
            content text
        );
        CREATE TABLE compound_pk (
            pk1 text,
            pk2 text,
            value text,
            PRIMARY KEY (pk1, pk2)
        );
        CREATE TABLE with_foreign_key (
            id integer primary key,
            fk_col integer references simple_primary_key(id)
        );
        CREATE VIEW my_view AS SELECT id, content FROM simple_primary_key;
        CREATE TABLE _hidden_table (id integer primary key);
        INSERT INTO simple_primary_key VALUES (1, 'hello');
        INSERT INTO simple_primary_key VALUES (2, 'world');
    """)
    # Keep write connection open so shared memory DB stays alive
    yield backend
    conn.close()


class TestRowProtocol:
    def test_sqlite_row_satisfies_protocol(self):
        from datasette.backends.base import RowProtocol

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("create table t (id integer primary key, name text)")
        conn.execute("insert into t values (1, 'hello')")
        row = conn.execute("select * from t").fetchone()

        assert isinstance(row, RowProtocol)
        # Integer indexing
        assert row[0] == 1
        assert row[1] == "hello"
        # String key access
        assert row["id"] == 1
        assert row["name"] == "hello"
        # keys()
        assert list(row.keys()) == ["id", "name"]
        # len
        assert len(row) == 2
        # iter
        assert list(row) == [1, "hello"]
        conn.close()


class TestDatabaseBackendABC:
    def test_cannot_instantiate(self):
        from datasette.backends.base import DatabaseBackend

        with pytest.raises(TypeError):
            DatabaseBackend()

    def test_has_backend_type(self):
        from datasette.backends.base import DatabaseBackend

        assert hasattr(DatabaseBackend, "backend_type")

    def test_column_namedtuple(self):
        from datasette.backends.base import Column

        col = Column(
            cid=0,
            name="id",
            type="integer",
            notnull=1,
            default_value=None,
            is_pk=1,
            hidden=0,
        )
        assert col.name == "id"
        assert col.is_pk == 1
        assert col._asdict() == {
            "cid": 0,
            "name": "id",
            "type": "integer",
            "notnull": 1,
            "default_value": None,
            "is_pk": 1,
            "hidden": 0,
        }


class TestSQLiteBackend:
    def test_backend_type(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True)
        assert backend.backend_type == "sqlite"

    def test_create_connection(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True, memory_name="test_conn")
        conn = backend.create_connection()
        assert conn is not None
        conn.execute("select 1")
        conn.close()

    def test_escape_identifier(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True)
        # Simple names pass through
        assert backend.escape_identifier("id") == "id"
        # Reserved words get brackets
        assert backend.escape_identifier("select") == "[select]"
        # Names with spaces get brackets
        assert backend.escape_identifier("my column") == "[my column]"

    def test_table_names(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        names = backend.table_names(conn)
        assert "simple_primary_key" in names
        assert "compound_pk" in names
        assert "with_foreign_key" in names
        assert "_hidden_table" in names
        # Views should NOT appear in table_names
        assert "my_view" not in names
        conn.close()

    def test_view_names(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        names = backend.view_names(conn)
        assert "my_view" in names
        assert "simple_primary_key" not in names
        conn.close()

    def test_table_exists(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        assert backend.table_exists(conn, "simple_primary_key") is True
        assert backend.table_exists(conn, "nonexistent") is False
        conn.close()

    def test_view_exists(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        assert backend.view_exists(conn, "my_view") is True
        assert backend.view_exists(conn, "nonexistent") is False
        conn.close()

    def test_table_columns(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        cols = backend.table_columns(conn, "simple_primary_key")
        assert cols == ["id", "content"]
        conn.close()

    def test_table_column_details(self, sqlite_backend_with_tables):
        from datasette.backends.base import Column

        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        details = backend.table_column_details(conn, "simple_primary_key")
        assert len(details) == 2
        assert isinstance(details[0], Column)
        assert details[0].name == "id"
        assert details[0].is_pk == 1
        assert details[1].name == "content"
        assert details[1].is_pk == 0
        conn.close()

    def test_primary_keys(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        assert backend.primary_keys(conn, "simple_primary_key") == ["id"]
        assert backend.primary_keys(conn, "compound_pk") == ["pk1", "pk2"]
        conn.close()

    def test_foreign_keys_for_table(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        fks = backend.foreign_keys_for_table(conn, "with_foreign_key")
        assert len(fks) == 1
        assert fks[0]["column"] == "fk_col"
        assert fks[0]["other_table"] == "simple_primary_key"
        assert fks[0]["other_column"] == "id"
        conn.close()

    def test_get_all_foreign_keys(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        all_fks = backend.get_all_foreign_keys(conn)
        assert "with_foreign_key" in all_fks
        assert len(all_fks["with_foreign_key"]["outgoing"]) == 1
        assert len(all_fks["simple_primary_key"]["incoming"]) == 1
        conn.close()

    def test_hidden_table_names(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        hidden = backend.hidden_table_names(conn)
        assert "_hidden_table" in hidden
        conn.close()

    def test_get_table_definition(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        defn = backend.get_table_definition(conn, "simple_primary_key")
        assert defn is not None
        assert "CREATE TABLE" in defn
        assert "simple_primary_key" in defn
        conn.close()

    def test_get_view_definition(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        defn = backend.get_view_definition(conn, "my_view")
        assert defn is not None
        assert "my_view" in defn
        conn.close()

    def test_detect_fts(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        # No FTS table for simple_primary_key
        assert backend.detect_fts(conn, "simple_primary_key") is None
        conn.close()

    def test_supports_fts(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True)
        assert backend.supports_fts() is True

    def test_translate_sql_noop(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True)
        sql = "select * from t where id = :id"
        assert backend.translate_sql(sql) == sql

    def test_suggest_name_from_path(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(path="/tmp/my_database.db")
        assert backend.suggest_name() == "my_database"

    def test_suggest_name_from_memory_name(self):
        from datasette.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(is_memory=True, memory_name="fixtures")
        assert backend.suggest_name() == "fixtures"

    def test_indexes_for_table(self, sqlite_backend_with_tables):
        backend = sqlite_backend_with_tables
        conn = backend.create_connection()
        indexes = backend.indexes_for_table(conn, "simple_primary_key")
        assert isinstance(indexes, list)
        conn.close()


class TestDatabaseDelegation:
    """Test that Database delegates to its backend."""

    @pytest_asyncio.fixture
    async def db_with_backend(self):
        from datasette.app import Datasette
        from datasette.database import Database
        from datasette.backends.sqlite import SQLiteBackend
        import secrets

        ds = Datasette(settings={"num_sql_threads": 1})
        name = f"test_delegation_{secrets.token_hex(4)}"
        backend = SQLiteBackend(ds=ds, is_memory=True, memory_name=name)
        db = ds.add_database(Database(ds, backend=backend), name="test_db")

        def prepare(conn):
            conn.executescript("""
                CREATE TABLE test_table (
                    id integer primary key,
                    name text not null
                );
                CREATE TABLE other_table (
                    id integer primary key,
                    test_id integer references test_table(id)
                );
                CREATE VIEW test_view AS SELECT id, name FROM test_table;
                INSERT INTO test_table VALUES (1, 'alpha');
                INSERT INTO test_table VALUES (2, 'beta');
            """)

        await db.execute_write_fn(prepare)
        return db

    @pytest.mark.asyncio
    async def test_has_backend(self, db_with_backend):
        db = db_with_backend
        from datasette.backends.sqlite import SQLiteBackend

        assert hasattr(db, "backend")
        assert isinstance(db.backend, SQLiteBackend)

    @pytest.mark.asyncio
    async def test_table_names(self, db_with_backend):
        db = db_with_backend
        names = await db.table_names()
        assert "test_table" in names
        assert "other_table" in names

    @pytest.mark.asyncio
    async def test_view_names(self, db_with_backend):
        db = db_with_backend
        names = await db.view_names()
        assert "test_view" in names

    @pytest.mark.asyncio
    async def test_table_exists(self, db_with_backend):
        db = db_with_backend
        assert await db.table_exists("test_table") is True
        assert await db.table_exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_view_exists(self, db_with_backend):
        db = db_with_backend
        assert await db.view_exists("test_view") is True
        assert await db.view_exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_table_columns(self, db_with_backend):
        db = db_with_backend
        cols = await db.table_columns("test_table")
        assert cols == ["id", "name"]

    @pytest.mark.asyncio
    async def test_primary_keys(self, db_with_backend):
        db = db_with_backend
        pks = await db.primary_keys("test_table")
        assert pks == ["id"]

    @pytest.mark.asyncio
    async def test_execute(self, db_with_backend):
        db = db_with_backend
        results = await db.execute("select id, name from test_table order by id")
        assert len(results.rows) == 2
        assert results.rows[0]["id"] == 1
        assert results.rows[0]["name"] == "alpha"

    @pytest.mark.asyncio
    async def test_escape_identifier(self, db_with_backend):
        db = db_with_backend
        assert db.escape_identifier("select") == "[select]"
        assert db.escape_identifier("id") == "id"

    @pytest.mark.asyncio
    async def test_backward_compat_no_backend(self):
        """Database(ds, memory_name=...) still works without explicit backend."""
        from datasette.app import Datasette
        from datasette.database import Database
        import secrets

        ds = Datasette(settings={"num_sql_threads": 1})
        name = f"test_compat_{secrets.token_hex(4)}"
        db = ds.add_database(Database(ds, memory_name=name), name="compat")

        def prepare(conn):
            conn.execute("create table foo (id integer primary key)")
            conn.execute("insert into foo values (42)")

        await db.execute_write_fn(prepare)
        results = await db.execute("select * from foo")
        assert results.rows[0][0] == 42


# ---- Plugin Hook and Backend Registry Tests ----


class TestBackendRegistry:
    @pytest.mark.asyncio
    async def test_builtin_backends_registered(self):
        from datasette.app import Datasette

        ds = Datasette()
        assert "sqlite" in ds._backend_registry

    @pytest.mark.asyncio
    async def test_hookspec_exists(self):
        from datasette.hookspecs import register_database_backends

        assert callable(register_database_backends)


class TestCLIConnectionStrings:
    def test_cli_connection_string_unknown_scheme(self):
        from click.testing import CliRunner
        from datasette.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["serve", "mysql://localhost/testdb", "--get", "/"]
        )
        assert result.exit_code != 0
        assert "Unknown database backend scheme" in result.output

    def test_cli_mixed_files_and_connection_strings(self, tmp_path):
        """SQLite files and connection strings can be mixed."""
        from click.testing import CliRunner
        from datasette.cli import cli

        # Create a test SQLite file
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("create table t (id integer primary key)")
        conn.close()

        runner = CliRunner()
        # This should fail because mysql:// is unknown, but that proves
        # it correctly separates SQLite files from connection strings
        result = runner.invoke(
            cli,
            ["serve", str(db_path), "mysql://localhost/db", "--get", "/"],
        )
        assert result.exit_code != 0
        assert "Unknown database backend scheme" in result.output
