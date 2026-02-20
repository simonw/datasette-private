"""Backend conformance tests using datasette.testing.

These run the shared conformance suite against the PostgreSQL backend.
Requires a PostgreSQL server. Set DATASETTE_TEST_POSTGRESQL=postgresql://...
to enable these tests.
"""

import os
import secrets

import pytest
import pytest_asyncio

from datasette.app import Datasette
from datasette.database import Database
from datasette.testing import (
    BackendIntrospectionTests,
    BackendExecutionTests,
    BackendHTTPTests,
    BackendWriteAPITests,
    EXPECTED_SCHEMA_SQL,
)
from datasette_postgresql.backend import PostgresBackend


POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")

requires_postgresql = pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL=postgresql://... to run PostgreSQL tests",
)

# Standard schema translated for PostgreSQL (no executescript, uses ; separation)
PG_SCHEMA_SQL = EXPECTED_SCHEMA_SQL


# ---- Fixtures for PostgreSQL ----


@pytest.fixture
def pg_schema():
    """Create a fresh PostgreSQL schema for isolation."""
    if not POSTGRESQL_TEST_URL:
        pytest.skip("No PostgreSQL configured")
    import psycopg

    schema_name = f"test_conform_{secrets.token_hex(4)}"
    conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema_name}")
    conn.execute(f"SET search_path TO {schema_name}")
    conn.execute(PG_SCHEMA_SQL)
    conn.close()
    yield schema_name
    cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cleanup_conn.close()


@pytest.fixture
def backend_with_tables(pg_schema):
    """Provide (backend, conn) for introspection tests."""
    backend = PostgresBackend(
        connection_string=POSTGRESQL_TEST_URL,
        schema=pg_schema,
        statement_timeout_ms=5000,
    )
    conn = backend.create_connection(write=True)
    yield backend, conn
    backend.close_connection(conn)


@pytest_asyncio.fixture
async def ds_db_with_tables(pg_schema):
    """Provide a Database instance for execution tests."""
    ds = Datasette(settings={"num_sql_threads": 3})
    backend = PostgresBackend(
        ds=ds,
        connection_string=POSTGRESQL_TEST_URL,
        schema=pg_schema,
        statement_timeout_ms=5000,
    )
    db = ds.add_database(Database(ds, backend=backend), name="testdb")
    return db


@pytest_asyncio.fixture
async def ds_client_with_tables(pg_schema):
    """Provide a Datasette client for HTTP tests."""
    ds = Datasette(settings={"num_sql_threads": 3})
    backend = PostgresBackend(
        ds=ds,
        connection_string=POSTGRESQL_TEST_URL,
        schema=pg_schema,
        statement_timeout_ms=5000,
    )
    ds.add_database(Database(ds, backend=backend), name="testdb")
    await ds.invoke_startup()
    return ds.client


@pytest_asyncio.fixture
async def ds_write():
    """Provide a Datasette instance for write API tests."""
    if not POSTGRESQL_TEST_URL:
        pytest.skip("No PostgreSQL configured")
    import psycopg

    schema_name = f"test_write_{secrets.token_hex(4)}"
    conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema_name}")
    conn.execute(f"SET search_path TO {schema_name}")
    conn.execute(
        "CREATE TABLE docs (id serial primary key, title text, score float, age integer)"
    )
    conn.close()

    ds = Datasette(settings={"num_sql_threads": 1})
    backend = PostgresBackend(
        ds=ds,
        connection_string=POSTGRESQL_TEST_URL,
        schema=schema_name,
        statement_timeout_ms=5000,
    )
    db = ds.add_database(
        Database(ds, backend=backend, is_mutable=True), name="data"
    )
    ds.root_enabled = True
    await ds.invoke_startup()
    yield ds

    backend.close_all()
    cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cleanup_conn.close()


# ---- Run conformance suites against PostgreSQL ----


@requires_postgresql
class TestPostgreSQLIntrospection(BackendIntrospectionTests):
    pass


@requires_postgresql
class TestPostgreSQLExecution(BackendExecutionTests):
    pass


@requires_postgresql
class TestPostgreSQLHTTP(BackendHTTPTests):
    pass


@requires_postgresql
class TestPostgreSQLWriteAPI(BackendWriteAPITests):
    pass
