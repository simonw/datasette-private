"""Backend conformance tests using datasette.testing.

These run the shared conformance suite against the SQLite backend.
"""

import secrets

import pytest
import pytest_asyncio

from datasette.app import Datasette
from datasette.backends.sqlite import SQLiteBackend
from datasette.database import Database
from datasette.testing import (
    BackendIntrospectionTests,
    BackendExecutionTests,
    BackendHTTPTests,
    BackendWriteAPITests,
    EXPECTED_SCHEMA_SQL,
)


# ---- Fixtures for SQLite ----


@pytest.fixture
def backend_with_tables():
    """Create a SQLiteBackend with the standard conformance schema."""
    name = f"test_conform_{secrets.token_hex(4)}"
    backend = SQLiteBackend(is_memory=True, memory_name=name)
    conn = backend.create_connection(write=True)
    conn.executescript(EXPECTED_SCHEMA_SQL)
    yield backend, conn
    conn.close()


@pytest_asyncio.fixture
async def ds_db_with_tables():
    """Create a Datasette + Database with the standard conformance schema."""
    name = f"test_exec_{secrets.token_hex(4)}"
    ds = Datasette(settings={"num_sql_threads": 1})
    backend = SQLiteBackend(ds=ds, is_memory=True, memory_name=name)
    db = ds.add_database(Database(ds, backend=backend), name="testdb")

    def setup(conn):
        conn.executescript(EXPECTED_SCHEMA_SQL)

    await db.execute_write_fn(setup)
    return db


@pytest_asyncio.fixture
async def ds_client_with_tables():
    """Create a Datasette client with the standard conformance schema."""
    name = f"test_http_{secrets.token_hex(4)}"
    ds = Datasette(settings={"num_sql_threads": 1})
    backend = SQLiteBackend(ds=ds, is_memory=True, memory_name=name)
    db = ds.add_database(Database(ds, backend=backend), name="testdb")

    def setup(conn):
        conn.executescript(EXPECTED_SCHEMA_SQL)

    await db.execute_write_fn(setup)
    await ds.invoke_startup()
    return ds.client


@pytest_asyncio.fixture
async def ds_write():
    """Create a Datasette instance with a mutable database for write tests."""
    name = f"test_write_{secrets.token_hex(4)}"
    ds = Datasette(settings={"num_sql_threads": 1})
    backend = SQLiteBackend(ds=ds, is_memory=True, memory_name=name)
    db = ds.add_database(
        Database(ds, backend=backend, is_mutable=True), name="data"
    )

    def setup(conn):
        conn.execute(
            "CREATE TABLE docs (id integer primary key, title text, score real, age integer)"
        )

    await db.execute_write_fn(setup)
    ds.root_enabled = True
    await ds.invoke_startup()
    return ds


# ---- Run conformance suites against SQLite ----


class TestSQLiteIntrospection(BackendIntrospectionTests):
    pass


class TestSQLiteExecution(BackendExecutionTests):
    pass


class TestSQLiteHTTP(BackendHTTPTests):
    pass


class TestSQLiteWriteAPI(BackendWriteAPITests):
    pass
