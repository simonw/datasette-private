"""Tests for the pluggable database backend system."""

import os
import pytest
import pytest_asyncio
import sqlite3


# PostgreSQL test support
POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")
requires_postgresql = pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL=postgresql://... to run PostgreSQL tests",
)


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
    async def test_backend_registry_has_postgresql_when_preview(self):
        from datasette.app import Datasette

        ds = Datasette(preview=True)
        try:
            import psycopg  # noqa: F401

            assert "postgresql" in ds._backend_registry
        except ImportError:
            assert "postgresql" not in ds._backend_registry

    @pytest.mark.asyncio
    async def test_backend_registry_no_postgresql_without_preview(self):
        from datasette.app import Datasette

        ds = Datasette()
        assert "postgresql" not in ds._backend_registry

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

    @requires_postgresql
    def test_cli_connection_string_postgresql(self):
        from click.testing import CliRunner
        from datasette.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "serve",
                "--preview",
                POSTGRESQL_TEST_URL,
                "--get",
                "/.json",
            ],
        )
        assert result.exit_code == 0, result.output
        import json

        data = json.loads(result.output)
        # Should have a database with the name from the connection string
        assert len(data.get("databases", [])) > 0

    @requires_postgresql
    def test_cli_connection_string_postgresql_requires_preview(self):
        from click.testing import CliRunner
        from datasette.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "serve",
                POSTGRESQL_TEST_URL,
                "--get",
                "/.json",
            ],
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


# ---- PostgresBackend Tests ----


@pytest.fixture(scope="module")
def pg_test_schema():
    """Create a test schema in PostgreSQL for isolation, tear down after."""
    if not POSTGRESQL_TEST_URL:
        pytest.skip("No PostgreSQL configured")
    import psycopg

    schema_name = f"test_datasette_{os.getpid()}"
    conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema_name}")
    conn.execute(
        f"SET search_path TO {schema_name}"
    )
    conn.execute("""
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
        INSERT INTO simple_primary_key VALUES (1, 'hello');
        INSERT INTO simple_primary_key VALUES (2, 'world');
    """)
    conn.close()
    yield schema_name
    # Cleanup
    cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cleanup_conn.close()


@pytest.fixture
def pg_backend(pg_test_schema):
    """Create a PostgresBackend connected to the test schema."""
    from datasette.backends.postgresql import PostgresBackend

    backend = PostgresBackend(
        connection_string=POSTGRESQL_TEST_URL,
        schema=pg_test_schema,
        statement_timeout_ms=5000,
    )
    yield backend


@requires_postgresql
class TestPostgresBackend:
    def test_backend_type(self, pg_backend):
        assert pg_backend.backend_type == "postgresql"

    def test_create_connection(self, pg_backend):
        conn = pg_backend.create_connection()
        assert conn is not None
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
        pg_backend.close_connection(conn)

    def test_create_read_only_connection(self, pg_backend):
        """Read connections should be read-only."""
        import psycopg

        conn = pg_backend.create_connection(write=False)
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute(
                "CREATE TABLE test_readonly_check (id integer primary key)"
            )
        pg_backend.close_connection(conn)

    def test_create_write_connection(self, pg_backend):
        """Write connections should allow writes."""
        conn = pg_backend.create_connection(write=True)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS test_write_check (id integer primary key)"
        )
        conn.execute("DROP TABLE IF EXISTS test_write_check")
        pg_backend.close_connection(conn)

    def test_escape_identifier(self, pg_backend):
        # Simple names pass through
        assert pg_backend.escape_identifier("id") == "id"
        # Reserved words get double-quoted
        assert pg_backend.escape_identifier("select") == '"select"'
        # Names with spaces get double-quoted
        assert pg_backend.escape_identifier("my column") == '"my column"'

    def test_translate_sql_params(self, pg_backend):
        """Convert :name params to %(name)s for psycopg."""
        sql = "select * from t where id = :id and name = :name"
        translated = pg_backend.translate_sql(sql)
        assert translated == "select * from t where id = %(id)s and name = %(name)s"

    def test_translate_sql_preserves_casts(self, pg_backend):
        """PostgreSQL :: casts should not be mangled."""
        sql = "select value::text from t where id = :id"
        translated = pg_backend.translate_sql(sql)
        assert "::text" in translated
        assert "%(id)s" in translated

    def test_table_names(self, pg_backend):
        conn = pg_backend.create_connection()
        names = pg_backend.table_names(conn)
        assert "simple_primary_key" in names
        assert "compound_pk" in names
        assert "with_foreign_key" in names
        # Views should NOT appear in table_names
        assert "my_view" not in names
        pg_backend.close_connection(conn)

    def test_view_names(self, pg_backend):
        conn = pg_backend.create_connection()
        names = pg_backend.view_names(conn)
        assert "my_view" in names
        assert "simple_primary_key" not in names
        pg_backend.close_connection(conn)

    def test_table_exists(self, pg_backend):
        conn = pg_backend.create_connection()
        assert pg_backend.table_exists(conn, "simple_primary_key") is True
        assert pg_backend.table_exists(conn, "nonexistent") is False
        pg_backend.close_connection(conn)

    def test_view_exists(self, pg_backend):
        conn = pg_backend.create_connection()
        assert pg_backend.view_exists(conn, "my_view") is True
        assert pg_backend.view_exists(conn, "nonexistent") is False
        pg_backend.close_connection(conn)

    def test_table_columns(self, pg_backend):
        conn = pg_backend.create_connection()
        cols = pg_backend.table_columns(conn, "simple_primary_key")
        assert cols == ["id", "content"]
        pg_backend.close_connection(conn)

    def test_table_column_details(self, pg_backend):
        from datasette.backends.base import Column

        conn = pg_backend.create_connection()
        details = pg_backend.table_column_details(conn, "simple_primary_key")
        assert len(details) == 2
        assert isinstance(details[0], Column)
        assert details[0].name == "id"
        assert details[0].is_pk == 1
        assert details[1].name == "content"
        assert details[1].is_pk == 0
        pg_backend.close_connection(conn)

    def test_primary_keys(self, pg_backend):
        conn = pg_backend.create_connection()
        assert pg_backend.primary_keys(conn, "simple_primary_key") == ["id"]
        assert pg_backend.primary_keys(conn, "compound_pk") == ["pk1", "pk2"]
        pg_backend.close_connection(conn)

    def test_foreign_keys_for_table(self, pg_backend):
        conn = pg_backend.create_connection()
        fks = pg_backend.foreign_keys_for_table(conn, "with_foreign_key")
        assert len(fks) == 1
        assert fks[0]["column"] == "fk_col"
        assert fks[0]["other_table"] == "simple_primary_key"
        assert fks[0]["other_column"] == "id"
        pg_backend.close_connection(conn)

    def test_get_all_foreign_keys(self, pg_backend):
        conn = pg_backend.create_connection()
        all_fks = pg_backend.get_all_foreign_keys(conn)
        assert "with_foreign_key" in all_fks
        assert len(all_fks["with_foreign_key"]["outgoing"]) == 1
        assert len(all_fks["simple_primary_key"]["incoming"]) == 1
        pg_backend.close_connection(conn)

    def test_hidden_table_names(self, pg_backend):
        """PostgreSQL has no shadow tables - hidden should be empty."""
        conn = pg_backend.create_connection()
        hidden = pg_backend.hidden_table_names(conn)
        assert hidden == []
        pg_backend.close_connection(conn)

    def test_get_table_definition(self, pg_backend):
        conn = pg_backend.create_connection()
        defn = pg_backend.get_table_definition(conn, "simple_primary_key")
        assert defn is not None
        assert "simple_primary_key" in defn
        pg_backend.close_connection(conn)

    def test_get_view_definition(self, pg_backend):
        conn = pg_backend.create_connection()
        defn = pg_backend.get_view_definition(conn, "my_view")
        assert defn is not None
        assert "my_view" in defn
        pg_backend.close_connection(conn)

    def test_detect_fts(self, pg_backend):
        """PostgreSQL does not support SQLite-style FTS."""
        conn = pg_backend.create_connection()
        assert pg_backend.detect_fts(conn, "simple_primary_key") is None
        pg_backend.close_connection(conn)

    def test_supports_fts(self, pg_backend):
        assert pg_backend.supports_fts() is False

    def test_suggest_name(self):
        from datasette.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            connection_string="postgresql://user:pass@localhost/salesdb"
        )
        assert backend.suggest_name() == "salesdb"

    def test_suggest_name_from_path(self):
        from datasette.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            connection_string="postgresql://localhost/my_data"
        )
        assert backend.suggest_name() == "my_data"

    def test_indexes_for_table(self, pg_backend):
        conn = pg_backend.create_connection()
        indexes = pg_backend.indexes_for_table(conn, "simple_primary_key")
        assert isinstance(indexes, list)
        # PostgreSQL creates an index for the primary key
        assert len(indexes) >= 1
        pg_backend.close_connection(conn)

    def test_row_satisfies_protocol(self, pg_backend):
        """PostgreSQL rows should satisfy RowProtocol."""
        from datasette.backends.base import RowProtocol

        conn = pg_backend.create_connection()
        cursor = conn.execute(
            "SELECT id, content FROM simple_primary_key ORDER BY id LIMIT 1"
        )
        row = cursor.fetchone()
        assert isinstance(row, RowProtocol)
        # Integer indexing
        assert row[0] == 1
        assert row[1] == "hello"
        # String key access
        assert row["id"] == 1
        assert row["content"] == "hello"
        # keys()
        assert list(row.keys()) == ["id", "content"]
        # len
        assert len(row) == 2
        # iter
        assert list(row) == [1, "hello"]
        pg_backend.close_connection(conn)

    def test_statement_timeout(self, pg_backend):
        """Statement timeout should cancel long-running queries."""
        import psycopg

        conn = pg_backend.create_connection()
        # pg_sleep(10) should be interrupted by the statement timeout
        with pytest.raises(psycopg.errors.QueryCanceled):
            conn.execute("SELECT pg_sleep(10)")
        pg_backend.close_connection(conn)


@requires_postgresql
class TestPostgresBackendAsync:
    """Test async execution paths of PostgresBackend."""

    @pytest_asyncio.fixture
    async def pg_ds_db(self, pg_test_schema):
        from datasette.app import Datasette
        from datasette.database import Database
        from datasette.backends.postgresql import PostgresBackend

        ds = Datasette(settings={"num_sql_threads": 3})
        backend = PostgresBackend(
            ds=ds,
            connection_string=POSTGRESQL_TEST_URL,
            schema=pg_test_schema,
        )
        db = ds.add_database(Database(ds, backend=backend), name="pgtest")
        yield db

    @pytest.mark.asyncio
    async def test_execute(self, pg_ds_db):
        db = pg_ds_db
        results = await db.execute(
            "select id, content from simple_primary_key order by id"
        )
        assert len(results.rows) == 2
        assert results.rows[0]["id"] == 1
        assert results.rows[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_with_params(self, pg_ds_db):
        db = pg_ds_db
        # Uses :name param style which gets translated to %(name)s
        results = await db.execute(
            "select id, content from simple_primary_key where id = :id",
            {"id": 1},
        )
        assert len(results.rows) == 1
        assert results.rows[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_fn(self, pg_ds_db):
        db = pg_ds_db

        def fn(conn):
            cursor = conn.execute("SELECT count(*) FROM simple_primary_key")
            return cursor.fetchone()[0]

        count = await db.execute_fn(fn)
        assert count == 2

    @pytest.mark.asyncio
    async def test_execute_write_fn(self, pg_ds_db):
        """PostgresBackend writes directly - no queue."""
        db = pg_ds_db

        def fn(conn):
            conn.execute(
                "CREATE TABLE IF NOT EXISTS test_write_async (id integer primary key, val text)"
            )
            conn.execute("INSERT INTO test_write_async VALUES (1, 'written')")

        await db.execute_write_fn(fn)

        results = await db.execute("SELECT val FROM test_write_async WHERE id = 1")
        assert results.rows[0]["val"] == "written"

        # Cleanup
        await db.execute_write_fn(
            lambda conn: conn.execute("DROP TABLE IF EXISTS test_write_async")
        )

    @pytest.mark.asyncio
    async def test_table_names_via_database(self, pg_ds_db):
        db = pg_ds_db
        names = await db.table_names()
        assert "simple_primary_key" in names

    @pytest.mark.asyncio
    async def test_table_columns_via_database(self, pg_ds_db):
        db = pg_ds_db
        cols = await db.table_columns("simple_primary_key")
        assert cols == ["id", "content"]

    @pytest.mark.asyncio
    async def test_primary_keys_via_database(self, pg_ds_db):
        db = pg_ds_db
        pks = await db.primary_keys("simple_primary_key")
        assert pks == ["id"]


@requires_postgresql
class TestPostgresBackendAsyncMethods:
    """Test all 16 async_* methods on PostgresBackend directly.

    Each test calls the async method and verifies it returns the same
    result as the corresponding sync method using a sync connection.
    """

    @pytest.fixture
    def pg_sync_conn(self, pg_backend):
        """Provide a sync connection from the backend, closed after the test."""
        conn = pg_backend.create_connection()
        yield conn
        pg_backend.close_connection(conn)

    # -- async_table_names --

    @pytest.mark.asyncio
    async def test_async_table_names(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_names()
        sync_result = pg_backend.table_names(pg_sync_conn)
        assert async_result == sync_result
        assert "simple_primary_key" in async_result
        assert "compound_pk" in async_result
        assert "with_foreign_key" in async_result
        # Views should NOT appear
        assert "my_view" not in async_result

    # -- async_view_names --

    @pytest.mark.asyncio
    async def test_async_view_names(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_view_names()
        sync_result = pg_backend.view_names(pg_sync_conn)
        assert async_result == sync_result
        assert "my_view" in async_result
        assert "simple_primary_key" not in async_result

    # -- async_table_exists --

    @pytest.mark.asyncio
    async def test_async_table_exists_true(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_exists("simple_primary_key")
        sync_result = pg_backend.table_exists(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result is True

    @pytest.mark.asyncio
    async def test_async_table_exists_false(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_exists("nonexistent")
        sync_result = pg_backend.table_exists(pg_sync_conn, "nonexistent")
        assert async_result == sync_result
        assert async_result is False

    # -- async_view_exists --

    @pytest.mark.asyncio
    async def test_async_view_exists_true(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_view_exists("my_view")
        sync_result = pg_backend.view_exists(pg_sync_conn, "my_view")
        assert async_result == sync_result
        assert async_result is True

    @pytest.mark.asyncio
    async def test_async_view_exists_false(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_view_exists("nonexistent")
        sync_result = pg_backend.view_exists(pg_sync_conn, "nonexistent")
        assert async_result == sync_result
        assert async_result is False

    # -- async_table_column_details --

    @pytest.mark.asyncio
    async def test_async_table_column_details(self, pg_backend, pg_sync_conn):
        from datasette.backends.base import Column

        async_result = await pg_backend.async_table_column_details("simple_primary_key")
        sync_result = pg_backend.table_column_details(pg_sync_conn, "simple_primary_key")
        assert len(async_result) == len(sync_result)
        for async_col, sync_col in zip(async_result, sync_result):
            assert isinstance(async_col, Column)
            assert async_col == sync_col
        # Verify specific column properties
        assert async_result[0].name == "id"
        assert async_result[0].is_pk == 1
        assert async_result[1].name == "content"
        assert async_result[1].is_pk == 0

    @pytest.mark.asyncio
    async def test_async_table_column_details_compound_pk(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_column_details("compound_pk")
        sync_result = pg_backend.table_column_details(pg_sync_conn, "compound_pk")
        assert len(async_result) == len(sync_result)
        for async_col, sync_col in zip(async_result, sync_result):
            assert async_col == sync_col

    # -- async_table_columns --

    @pytest.mark.asyncio
    async def test_async_table_columns(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_columns("simple_primary_key")
        sync_result = pg_backend.table_columns(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result == ["id", "content"]

    @pytest.mark.asyncio
    async def test_async_table_columns_compound(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_table_columns("compound_pk")
        sync_result = pg_backend.table_columns(pg_sync_conn, "compound_pk")
        assert async_result == sync_result
        assert async_result == ["pk1", "pk2", "value"]

    # -- async_primary_keys --

    @pytest.mark.asyncio
    async def test_async_primary_keys_single(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_primary_keys("simple_primary_key")
        sync_result = pg_backend.primary_keys(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result == ["id"]

    @pytest.mark.asyncio
    async def test_async_primary_keys_compound(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_primary_keys("compound_pk")
        sync_result = pg_backend.primary_keys(pg_sync_conn, "compound_pk")
        assert async_result == sync_result
        assert async_result == ["pk1", "pk2"]

    # -- async_detect_fts --

    @pytest.mark.asyncio
    async def test_async_detect_fts(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_detect_fts("simple_primary_key")
        sync_result = pg_backend.detect_fts(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result is None

    # -- async_foreign_keys_for_table --

    @pytest.mark.asyncio
    async def test_async_foreign_keys_for_table(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_foreign_keys_for_table("with_foreign_key")
        sync_result = pg_backend.foreign_keys_for_table(pg_sync_conn, "with_foreign_key")
        assert async_result == sync_result
        assert len(async_result) == 1
        assert async_result[0]["column"] == "fk_col"
        assert async_result[0]["other_table"] == "simple_primary_key"
        assert async_result[0]["other_column"] == "id"

    @pytest.mark.asyncio
    async def test_async_foreign_keys_for_table_none(self, pg_backend, pg_sync_conn):
        """Table with no foreign keys should return empty list."""
        async_result = await pg_backend.async_foreign_keys_for_table("simple_primary_key")
        sync_result = pg_backend.foreign_keys_for_table(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result == []

    # -- async_get_all_foreign_keys --

    @pytest.mark.asyncio
    async def test_async_get_all_foreign_keys(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_get_all_foreign_keys()
        sync_result = pg_backend.get_all_foreign_keys(pg_sync_conn)
        assert async_result == sync_result
        # Verify structure
        assert "with_foreign_key" in async_result
        assert "simple_primary_key" in async_result
        assert len(async_result["with_foreign_key"]["outgoing"]) == 1
        assert len(async_result["simple_primary_key"]["incoming"]) == 1
        # Tables with no foreign keys should have empty lists
        assert async_result["compound_pk"]["incoming"] == []
        assert async_result["compound_pk"]["outgoing"] == []

    # -- async_hidden_table_names --

    @pytest.mark.asyncio
    async def test_async_hidden_table_names(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_hidden_table_names()
        sync_result = pg_backend.hidden_table_names(pg_sync_conn)
        assert async_result == sync_result
        assert async_result == []

    # -- async_get_table_definition --

    @pytest.mark.asyncio
    async def test_async_get_table_definition(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_get_table_definition("simple_primary_key")
        sync_result = pg_backend.get_table_definition(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert async_result is not None
        assert "CREATE TABLE" in async_result
        assert "simple_primary_key" in async_result

    @pytest.mark.asyncio
    async def test_async_get_table_definition_delegates_to_view(self, pg_backend, pg_sync_conn):
        """When type_='view', it should delegate to async_get_view_definition."""
        async_result = await pg_backend.async_get_table_definition("my_view", type_="view")
        sync_result = pg_backend.get_table_definition(pg_sync_conn, "my_view", type_="view")
        assert async_result == sync_result
        assert async_result is not None
        assert "my_view" in async_result

    @pytest.mark.asyncio
    async def test_async_get_table_definition_nonexistent(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_get_table_definition("nonexistent_table")
        sync_result = pg_backend.get_table_definition(pg_sync_conn, "nonexistent_table")
        assert async_result == sync_result
        assert async_result is None

    # -- async_get_view_definition --

    @pytest.mark.asyncio
    async def test_async_get_view_definition(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_get_view_definition("my_view")
        sync_result = pg_backend.get_view_definition(pg_sync_conn, "my_view")
        assert async_result == sync_result
        assert async_result is not None
        assert "my_view" in async_result
        assert "CREATE VIEW" in async_result

    @pytest.mark.asyncio
    async def test_async_get_view_definition_nonexistent(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_get_view_definition("nonexistent_view")
        sync_result = pg_backend.get_view_definition(pg_sync_conn, "nonexistent_view")
        assert async_result == sync_result
        assert async_result is None

    # -- async_indexes_for_table --

    @pytest.mark.asyncio
    async def test_async_indexes_for_table(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_indexes_for_table("simple_primary_key")
        sync_result = pg_backend.indexes_for_table(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert isinstance(async_result, list)
        # PostgreSQL creates an index for the primary key
        assert len(async_result) >= 1
        # Each index should have the expected keys
        for idx in async_result:
            assert "name" in idx
            assert "unique" in idx
            assert "sql" in idx

    @pytest.mark.asyncio
    async def test_async_indexes_for_table_compound_pk(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_indexes_for_table("compound_pk")
        sync_result = pg_backend.indexes_for_table(pg_sync_conn, "compound_pk")
        assert async_result == sync_result
        assert len(async_result) >= 1

    # -- async_label_column_details --

    @pytest.mark.asyncio
    async def test_async_label_column_details(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_label_column_details("simple_primary_key")
        sync_result = pg_backend.label_column_details(pg_sync_conn, "simple_primary_key")
        assert async_result == sync_result
        assert "id" in async_result
        assert "content" in async_result
        # "content" is a text column
        py_type, is_unique = async_result["content"]
        assert py_type is str
        # "id" is an integer column and is a unique primary key
        py_type_id, is_unique_id = async_result["id"]
        assert py_type_id is not str
        assert is_unique_id is True

    @pytest.mark.asyncio
    async def test_async_label_column_details_compound_pk(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_label_column_details("compound_pk")
        sync_result = pg_backend.label_column_details(pg_sync_conn, "compound_pk")
        assert async_result == sync_result
        assert "pk1" in async_result
        assert "pk2" in async_result
        assert "value" in async_result

    # -- async_schema_version --

    @pytest.mark.asyncio
    async def test_async_schema_version(self, pg_backend, pg_sync_conn):
        async_result = await pg_backend.async_schema_version()
        sync_result = pg_backend.schema_version(pg_sync_conn)
        assert async_result == sync_result
        assert isinstance(async_result, int)
        assert async_result > 0

    @pytest.mark.asyncio
    async def test_async_schema_version_changes_on_ddl(self, pg_backend):
        """Schema version should change when a table is added or dropped."""
        version_before = await pg_backend.async_schema_version()

        # Add a new table using a sync write connection
        write_conn = pg_backend.create_connection(write=True)
        write_conn.execute(
            "CREATE TABLE test_schema_version_change (id integer primary key)"
        )
        pg_backend.close_connection(write_conn)

        version_after = await pg_backend.async_schema_version()
        assert version_after != version_before

        # Clean up
        cleanup_conn = pg_backend.create_connection(write=True)
        cleanup_conn.execute("DROP TABLE IF EXISTS test_schema_version_change")
        pg_backend.close_connection(cleanup_conn)
