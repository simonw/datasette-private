"""Parameterized integration tests that run against both SQLite and PostgreSQL backends.

These test the Datasette HTTP API layer to verify that both backends produce
identical results for common operations.
"""

import os
import secrets

import pytest
import pytest_asyncio

from datasette.app import Datasette
from datasette.database import Database


POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")


def _make_sqlite_ds():
    """Create a Datasette with an in-memory SQLite backend."""
    name = f"test_integration_{secrets.token_hex(4)}"
    ds = Datasette(settings={"num_sql_threads": 1})
    db = ds.add_database(Database(ds, memory_name=name), name="testdb")
    return ds, db


def _make_pg_ds(pg_schema):
    """Create a Datasette with a PostgreSQL backend."""
    from datasette.backends.postgresql import PostgresBackend

    ds = Datasette(settings={"num_sql_threads": 1})
    backend = PostgresBackend(
        ds=ds,
        connection_string=POSTGRESQL_TEST_URL,
        schema=pg_schema,
    )
    db = ds.add_database(Database(ds, backend=backend), name="testdb")
    return ds, db


@pytest.fixture
def pg_integration_schema():
    """Create a fresh PostgreSQL schema for each test."""
    if not POSTGRESQL_TEST_URL:
        yield None
        return
    import psycopg

    schema_name = f"test_int_{secrets.token_hex(4)}"
    conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema_name}")
    conn.close()
    yield schema_name
    cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cleanup_conn.close()


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgresql",
            marks=pytest.mark.skipif(
                not POSTGRESQL_TEST_URL,
                reason="Set DATASETTE_TEST_POSTGRESQL to run PostgreSQL tests",
            ),
        ),
    ]
)
def backend_type(request):
    return request.param


@pytest_asyncio.fixture
async def ds_client(backend_type, pg_integration_schema):
    """Create a Datasette client with test data on the specified backend."""
    if backend_type == "sqlite":
        ds, db = _make_sqlite_ds()
    else:
        ds, db = _make_pg_ds(pg_integration_schema)

    # Populate test data using standard SQL that works on both backends
    async def setup_data():
        await db.execute_write(
            "CREATE TABLE IF NOT EXISTS simple_primary_key ("
            "id integer primary key, content text)"
        )
        await db.execute_write(
            "CREATE TABLE IF NOT EXISTS compound_pk ("
            "pk1 text, pk2 text, value text, PRIMARY KEY (pk1, pk2))"
        )
        for i, val in enumerate(["hello", "world", "test"], 1):
            await db.execute_write(
                "INSERT INTO simple_primary_key (id, content) VALUES (:id, :content)",
                {"id": i, "content": val},
            )
        await db.execute_write(
            "INSERT INTO compound_pk (pk1, pk2, value) VALUES (:pk1, :pk2, :value)",
            {"pk1": "a", "pk2": "b", "value": "ab_val"},
        )
        await db.execute_write(
            "INSERT INTO compound_pk (pk1, pk2, value) VALUES (:pk1, :pk2, :value)",
            {"pk1": "c", "pk2": "d", "value": "cd_val"},
        )

    await setup_data()
    await ds.invoke_startup()
    yield ds.client


class TestBothBackendsIntegration:
    @pytest.mark.asyncio
    async def test_homepage(self, ds_client):
        response = await ds_client.get("/.json")
        assert response.status_code == 200
        data = response.json()
        assert "testdb" in data["databases"]

    @pytest.mark.asyncio
    async def test_database_page(self, ds_client):
        response = await ds_client.get("/testdb.json")
        assert response.status_code == 200
        data = response.json()
        table_names = [t["name"] for t in data["tables"]]
        assert "simple_primary_key" in table_names
        assert "compound_pk" in table_names

    @pytest.mark.asyncio
    async def test_table_view(self, ds_client):
        response = await ds_client.get("/testdb/simple_primary_key.json?_shape=array")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["id"] == 1
        assert data[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_table_view_with_filter(self, ds_client):
        response = await ds_client.get(
            "/testdb/simple_primary_key.json?content=hello&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_table_view_with_sort(self, ds_client):
        response = await ds_client.get(
            "/testdb/simple_primary_key.json?_sort_desc=id&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["id"] == 3
        assert data[-1]["id"] == 1

    @pytest.mark.asyncio
    async def test_row_view(self, ds_client):
        response = await ds_client.get("/testdb/simple_primary_key/1.json")
        assert response.status_code == 200
        data = response.json()
        assert data["rows"][0]["id"] == 1
        assert data["rows"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_compound_pk_row(self, ds_client):
        response = await ds_client.get("/testdb/compound_pk/a,b.json")
        assert response.status_code == 200
        data = response.json()
        assert data["rows"][0]["pk1"] == "a"
        assert data["rows"][0]["pk2"] == "b"
        assert data["rows"][0]["value"] == "ab_val"

    @pytest.mark.asyncio
    async def test_arbitrary_sql(self, ds_client):
        response = await ds_client.get(
            "/testdb/-/query.json?sql=select+count(*)+as+cnt+from+simple_primary_key&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["cnt"] == 3

    @pytest.mark.asyncio
    async def test_table_rows(self, ds_client):
        response = await ds_client.get("/testdb/simple_primary_key.json")
        assert response.status_code == 200
        data = response.json()
        assert len(data["rows"]) == 3

    @pytest.mark.asyncio
    async def test_table_schema(self, ds_client):
        response = await ds_client.get("/testdb/simple_primary_key/-/schema.json")
        assert response.status_code == 200
        data = response.json()
        assert "simple_primary_key" in data.get("schema", "")

    @pytest.mark.asyncio
    async def test_pagination(self, ds_client):
        # Request with _size=1 to force pagination
        response = await ds_client.get(
            "/testdb/simple_primary_key.json?_size=1&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1


# ---- Mixed Backend Test ----


@pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL to run mixed backend tests",
)
class TestMixedBackends:
    """Test SQLite and PostgreSQL databases in the same Datasette instance."""

    @pytest_asyncio.fixture
    async def mixed_client(self):
        from datasette.backends.postgresql import PostgresBackend
        import psycopg

        # Create PostgreSQL schema
        pg_schema = f"test_mixed_{secrets.token_hex(4)}"
        conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
        conn.execute(f"DROP SCHEMA IF EXISTS {pg_schema} CASCADE")
        conn.execute(f"CREATE SCHEMA {pg_schema}")
        conn.execute(f"SET search_path TO {pg_schema}")
        conn.execute(
            "CREATE TABLE pg_data (id integer primary key, name text)"
        )
        conn.execute("INSERT INTO pg_data VALUES (1, 'from_postgres')")
        conn.close()

        # Create Datasette with both backends
        sqlite_name = f"test_mixed_{secrets.token_hex(4)}"
        ds = Datasette(settings={"num_sql_threads": 3})

        # Add SQLite database
        sqlite_db = ds.add_database(
            Database(ds, memory_name=sqlite_name), name="sqlite_db"
        )

        def populate_sqlite(conn):
            conn.execute(
                "CREATE TABLE local_data (id integer primary key, name text)"
            )
            conn.execute("INSERT INTO local_data VALUES (1, 'from_sqlite')")

        await sqlite_db.execute_write_fn(populate_sqlite)

        # Add PostgreSQL database
        pg_backend = PostgresBackend(
            ds=ds,
            connection_string=POSTGRESQL_TEST_URL,
            schema=pg_schema,
        )
        ds.add_database(Database(ds, backend=pg_backend), name="pg_db")

        await ds.invoke_startup()

        yield ds.client

        # Cleanup
        cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
        cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {pg_schema} CASCADE")
        cleanup_conn.close()

    @pytest.mark.asyncio
    async def test_homepage_shows_both(self, mixed_client):
        response = await mixed_client.get("/.json")
        assert response.status_code == 200
        data = response.json()
        assert "sqlite_db" in data["databases"]
        assert "pg_db" in data["databases"]

    @pytest.mark.asyncio
    async def test_local_database(self, mixed_client):
        response = await mixed_client.get(
            "/sqlite_db/local_data.json?_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "from_sqlite"

    @pytest.mark.asyncio
    async def test_pg_database(self, mixed_client):
        response = await mixed_client.get(
            "/pg_db/pg_data.json?_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "from_postgres"

    @pytest.mark.asyncio
    async def test_both_databases_in_same_request_cycle(self, mixed_client):
        """Verify both backends work within the same Datasette instance."""
        # Query SQLite
        sqlite_resp = await mixed_client.get(
            "/sqlite_db/local_data.json?_shape=array"
        )
        assert sqlite_resp.status_code == 200

        # Query PostgreSQL
        pg_resp = await mixed_client.get(
            "/pg_db/pg_data.json?_shape=array"
        )
        assert pg_resp.status_code == 200

        # Verify they return different data from different backends
        local_data = sqlite_resp.json()
        pg_data = pg_resp.json()
        assert local_data[0]["name"] == "from_sqlite"
        assert pg_data[0]["name"] == "from_postgres"

    @pytest.mark.asyncio
    async def test_schema_endpoint_both_backends(self, mixed_client):
        sqlite_resp = await mixed_client.get("/sqlite_db/-/schema.json")
        assert sqlite_resp.status_code == 200
        assert "local_data" in sqlite_resp.json().get("schema", "")

        pg_resp = await mixed_client.get("/pg_db/-/schema.json")
        assert pg_resp.status_code == 200
        assert "pg_data" in pg_resp.json().get("schema", "")
