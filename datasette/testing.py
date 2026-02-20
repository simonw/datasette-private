"""Shared conformance test suite for database backends.

Backend authors subclass these test classes and provide the required fixtures
in their conftest.py. Datasette's own test suite uses them for SQLite.

Required fixtures per test class:

BackendIntrospectionTests:
    backend_with_tables -> (backend, conn)
        A backend with the standard schema populated (see EXPECTED_SCHEMA_SQL).
        `conn` is a live connection for sync introspection calls.

BackendExecutionTests:
    ds_db_with_tables -> Database
        A Database instance (with backend) containing the standard schema.

BackendHTTPTests:
    ds_client_with_tables -> AsyncClient
        A Datasette HTTPX async client, database named "testdb".

BackendWriteAPITests:
    ds_write -> Datasette
        A Datasette instance with root_enabled=True, a mutable database
        named "data" containing a table:
            docs (id <auto-pk>, title text, score real/float, age integer)

Usage:

    from datasette.testing import BackendIntrospectionTests

    class TestMyBackendIntrospection(BackendIntrospectionTests):
        @pytest.fixture
        def backend_with_tables(self):
            ...
            yield backend, conn
"""

import pytest

# Standard schema that all conformance fixtures must create.
EXPECTED_SCHEMA_SQL = """
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
""".strip()


# ---- Introspection conformance ----


class BackendIntrospectionTests:
    """Conformance tests for sync introspection methods on a DatabaseBackend.

    Subclass this and provide a ``backend_with_tables`` fixture that yields
    ``(backend, conn)`` with the standard schema populated.
    """

    def test_table_names(self, backend_with_tables):
        backend, conn = backend_with_tables
        names = backend.table_names(conn)
        assert "simple_primary_key" in names
        assert "compound_pk" in names
        assert "with_foreign_key" in names
        # Views must NOT appear in table_names
        assert "my_view" not in names

    def test_view_names(self, backend_with_tables):
        backend, conn = backend_with_tables
        names = backend.view_names(conn)
        assert "my_view" in names
        assert "simple_primary_key" not in names

    def test_table_exists(self, backend_with_tables):
        backend, conn = backend_with_tables
        assert backend.table_exists(conn, "simple_primary_key") is True
        assert backend.table_exists(conn, "nonexistent") is False

    def test_view_exists(self, backend_with_tables):
        backend, conn = backend_with_tables
        assert backend.view_exists(conn, "my_view") is True
        assert backend.view_exists(conn, "nonexistent") is False

    def test_table_columns(self, backend_with_tables):
        backend, conn = backend_with_tables
        cols = backend.table_columns(conn, "simple_primary_key")
        assert cols == ["id", "content"]

    def test_table_column_details(self, backend_with_tables):
        from datasette.backends.base import Column

        backend, conn = backend_with_tables
        details = backend.table_column_details(conn, "simple_primary_key")
        assert len(details) == 2
        assert isinstance(details[0], Column)
        assert details[0].name == "id"
        assert details[0].is_pk == 1
        assert details[1].name == "content"
        assert details[1].is_pk == 0

    def test_primary_keys_single(self, backend_with_tables):
        backend, conn = backend_with_tables
        assert backend.primary_keys(conn, "simple_primary_key") == ["id"]

    def test_primary_keys_compound(self, backend_with_tables):
        backend, conn = backend_with_tables
        assert backend.primary_keys(conn, "compound_pk") == ["pk1", "pk2"]

    def test_foreign_keys_for_table(self, backend_with_tables):
        backend, conn = backend_with_tables
        fks = backend.foreign_keys_for_table(conn, "with_foreign_key")
        assert len(fks) == 1
        assert fks[0]["column"] == "fk_col"
        assert fks[0]["other_table"] == "simple_primary_key"
        assert fks[0]["other_column"] == "id"

    def test_foreign_keys_empty(self, backend_with_tables):
        backend, conn = backend_with_tables
        fks = backend.foreign_keys_for_table(conn, "simple_primary_key")
        assert fks == []

    def test_get_all_foreign_keys(self, backend_with_tables):
        backend, conn = backend_with_tables
        all_fks = backend.get_all_foreign_keys(conn)
        assert "with_foreign_key" in all_fks
        assert len(all_fks["with_foreign_key"]["outgoing"]) == 1
        assert len(all_fks["simple_primary_key"]["incoming"]) == 1

    def test_get_table_definition(self, backend_with_tables):
        backend, conn = backend_with_tables
        defn = backend.get_table_definition(conn, "simple_primary_key")
        assert defn is not None
        assert "simple_primary_key" in defn

    def test_get_view_definition(self, backend_with_tables):
        backend, conn = backend_with_tables
        defn = backend.get_view_definition(conn, "my_view")
        assert defn is not None
        assert "my_view" in defn

    def test_indexes_for_table(self, backend_with_tables):
        backend, conn = backend_with_tables
        indexes = backend.indexes_for_table(conn, "simple_primary_key")
        assert isinstance(indexes, list)

    def test_row_satisfies_protocol(self, backend_with_tables):
        from datasette.backends.base import RowProtocol

        backend, conn = backend_with_tables
        # Prepare the connection so row_factory is set
        backend.prepare_connection(conn, None, "")
        cursor = conn.execute(
            "SELECT id, content FROM simple_primary_key ORDER BY id LIMIT 1"
        )
        row = cursor.fetchone()
        assert isinstance(row, RowProtocol)
        assert row[0] == 1
        assert row[1] == "hello"
        assert row["id"] == 1
        assert row["content"] == "hello"
        assert list(row.keys()) == ["id", "content"]
        assert len(row) == 2
        assert list(row) == [1, "hello"]

    def test_escape_identifier_simple(self, backend_with_tables):
        backend, conn = backend_with_tables
        # Simple alphanumeric names should be usable as-is
        result = backend.escape_identifier("id")
        assert "id" in result  # Could be "id" or [id] or "id" depending on backend

    def test_escape_identifier_reserved(self, backend_with_tables):
        backend, conn = backend_with_tables
        result = backend.escape_identifier("select")
        # Must be quoted somehow
        assert result != "select"


# ---- Async execution conformance ----


class BackendExecutionTests:
    """Conformance tests for async execution through a Database object.

    Subclass this and provide a ``ds_db_with_tables`` fixture that returns
    a ``Database`` instance with the standard schema populated.
    """

    @pytest.mark.asyncio
    async def test_execute_basic(self, ds_db_with_tables):
        db = ds_db_with_tables
        results = await db.execute(
            "select id, content from simple_primary_key order by id"
        )
        assert len(results.rows) == 2
        assert results.rows[0]["id"] == 1
        assert results.rows[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_with_params(self, ds_db_with_tables):
        db = ds_db_with_tables
        results = await db.execute(
            "select id, content from simple_primary_key where id = :id",
            {"id": 1},
        )
        assert len(results.rows) == 1
        assert results.rows[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_fn(self, ds_db_with_tables):
        db = ds_db_with_tables

        def fn(conn):
            cursor = conn.execute("SELECT count(*) FROM simple_primary_key")
            return cursor.fetchone()[0]

        count = await db.execute_fn(fn)
        assert count == 2

    @pytest.mark.asyncio
    async def test_execute_write_fn(self, ds_db_with_tables):
        db = ds_db_with_tables

        def fn(conn):
            conn.execute(
                "CREATE TABLE IF NOT EXISTS test_write (id integer primary key, val text)"
            )
            conn.execute("INSERT INTO test_write VALUES (1, 'written')")

        await db.execute_write_fn(fn)

        results = await db.execute("SELECT val FROM test_write WHERE id = 1")
        assert results.rows[0]["val"] == "written"

        # Cleanup
        await db.execute_write_fn(
            lambda conn: conn.execute("DROP TABLE IF EXISTS test_write")
        )

    @pytest.mark.asyncio
    async def test_table_names_via_database(self, ds_db_with_tables):
        db = ds_db_with_tables
        names = await db.table_names()
        assert "simple_primary_key" in names

    @pytest.mark.asyncio
    async def test_table_columns_via_database(self, ds_db_with_tables):
        db = ds_db_with_tables
        cols = await db.table_columns("simple_primary_key")
        assert cols == ["id", "content"]

    @pytest.mark.asyncio
    async def test_primary_keys_via_database(self, ds_db_with_tables):
        db = ds_db_with_tables
        pks = await db.primary_keys("simple_primary_key")
        assert pks == ["id"]


# ---- HTTP API conformance ----


class BackendHTTPTests:
    """Conformance tests for the Datasette HTTP API.

    Subclass this and provide a ``ds_client_with_tables`` fixture that returns
    a Datasette ``AsyncClient``, with the database named "testdb".
    """

    @pytest.mark.asyncio
    async def test_homepage(self, ds_client_with_tables):
        response = await ds_client_with_tables.get("/.json")
        assert response.status_code == 200
        data = response.json()
        assert "testdb" in data["databases"]

    @pytest.mark.asyncio
    async def test_database_page(self, ds_client_with_tables):
        response = await ds_client_with_tables.get("/testdb.json")
        assert response.status_code == 200
        data = response.json()
        table_names = [t["name"] for t in data["tables"]]
        assert "simple_primary_key" in table_names
        assert "compound_pk" in table_names

    @pytest.mark.asyncio
    async def test_table_view(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/simple_primary_key.json?_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_table_view_with_filter(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/simple_primary_key.json?content=hello&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_table_view_with_sort(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/simple_primary_key.json?_sort_desc=id&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["id"] == 2
        assert data[-1]["id"] == 1

    @pytest.mark.asyncio
    async def test_row_view(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/simple_primary_key/1.json"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rows"][0]["id"] == 1
        assert data["rows"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_compound_pk_row(self, ds_client_with_tables):
        """Need compound_pk rows to be inserted by the fixture for this test."""
        # First insert some data
        pass  # Compound PK data not in the standard schema inserts

    @pytest.mark.asyncio
    async def test_arbitrary_sql(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/-/query.json?sql=select+count(*)+as+cnt+from+simple_primary_key&_shape=array"
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["cnt"] == 2

    @pytest.mark.asyncio
    async def test_table_schema(self, ds_client_with_tables):
        response = await ds_client_with_tables.get(
            "/testdb/simple_primary_key/-/schema.json"
        )
        assert response.status_code == 200
        data = response.json()
        assert "simple_primary_key" in data.get("schema", "")


# ---- Write API conformance ----


def _write_headers(ds):
    """Get auth headers for write operations."""
    token = ds.create_token("root")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


class BackendWriteAPITests:
    """Conformance tests for the Datasette write API.

    Subclass this and provide a ``ds_write`` fixture that returns a Datasette
    instance with root_enabled=True and a mutable database named "data"
    containing: docs (id <auto-pk>, title text, score real/float, age integer)
    """

    @pytest.mark.asyncio
    async def test_insert_row(self, ds_write):
        headers = _write_headers(ds_write)
        response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Test", "score": 1.2, "age": 5}},
            headers=headers,
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        rows = response.json()["rows"]
        assert len(rows) == 1
        assert rows[0]["title"] == "Test"
        assert rows[0]["score"] == 1.2
        assert rows[0]["age"] == 5

    @pytest.mark.asyncio
    async def test_insert_rows_bulk(self, ds_write):
        headers = _write_headers(ds_write)
        data = {
            "rows": [
                {"title": f"Test {i}", "score": 1.0, "age": 5} for i in range(20)
            ],
            "return": True,
        }
        response = await ds_write.client.post(
            "/data/docs/-/insert",
            json=data,
            headers=headers,
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        assert len(response.json()["rows"]) == 20

    @pytest.mark.asyncio
    async def test_insert_row_alter(self, ds_write):
        headers = _write_headers(ds_write)
        response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={
                "row": {"title": "Test", "score": 1.2, "age": 5, "extra": "extra"},
                "alter": True,
            },
            headers=headers,
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        assert response.json()["rows"][0]["extra"] == "extra"

    @pytest.mark.asyncio
    async def test_insert_ignore(self, ds_write):
        headers = _write_headers(ds_write)
        # Insert first row with explicit PK
        await ds_write.client.post(
            "/data/docs/-/insert",
            json={"row": {"id": 1, "title": "Original"}},
            headers=headers,
        )
        # Insert with ignore (should not overwrite)
        response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={"rows": [{"id": 1, "title": "Duplicate"}], "ignore": True},
            headers=headers,
        )
        assert response.status_code == 201
        row = (
            await ds_write.get_database("data").execute(
                "select title from docs where id = 1"
            )
        ).rows[0][0]
        assert row == "Original"

    @pytest.mark.asyncio
    async def test_insert_replace(self, ds_write):
        headers = _write_headers(ds_write)
        await ds_write.client.post(
            "/data/docs/-/insert",
            json={"row": {"id": 1, "title": "Original"}},
            headers=headers,
        )
        response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={"rows": [{"id": 1, "title": "Replaced"}], "replace": True},
            headers=headers,
        )
        assert response.status_code == 201
        row = (
            await ds_write.get_database("data").execute(
                "select title from docs where id = 1"
            )
        ).rows[0][0]
        assert row == "Replaced"

    @pytest.mark.asyncio
    async def test_upsert(self, ds_write):
        headers = _write_headers(ds_write)
        # Create a table with data
        await ds_write.client.post(
            "/data/-/create",
            json={
                "table": "upsert_test",
                "rows": [{"id": 1, "title": "One"}],
                "pk": "id",
            },
            headers=headers,
        )
        # Upsert: update existing + insert new
        response = await ds_write.client.post(
            "/data/upsert_test/-/upsert",
            json={
                "rows": [
                    {"id": 1, "title": "Updated One"},
                    {"id": 2, "title": "Two"},
                ],
            },
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        actual_rows = (
            await ds_write.client.get(
                "/data/upsert_test.json?_shape=array&_sort=id"
            )
        ).json()
        assert actual_rows == [
            {"id": 1, "title": "Updated One"},
            {"id": 2, "title": "Two"},
        ]

    @pytest.mark.asyncio
    async def test_delete_row(self, ds_write):
        headers = _write_headers(ds_write)
        insert_response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "To Delete", "score": 1.0, "age": 5}},
            headers=headers,
        )
        pk = insert_response.json()["rows"][0]["id"]
        response = await ds_write.client.post(
            f"/data/docs/{pk}/-/delete",
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        count = (
            await ds_write.get_database("data").execute(
                "select count(*) from docs"
            )
        ).rows[0][0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_row(self, ds_write):
        headers = _write_headers(ds_write)
        insert_response = await ds_write.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Original", "score": 1.0, "age": 5}},
            headers=headers,
        )
        pk = insert_response.json()["rows"][0]["id"]
        response = await ds_write.client.post(
            f"/data/docs/{pk}/-/update",
            json={"update": {"title": "Updated", "score": 2.5}, "return": True},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["row"]["title"] == "Updated"
        assert response.json()["row"]["score"] == 2.5
        assert response.json()["row"]["age"] == 5

    @pytest.mark.asyncio
    async def test_drop_table(self, ds_write):
        headers = _write_headers(ds_write)
        await ds_write.client.post(
            "/data/-/create",
            json={
                "table": "to_drop",
                "rows": [{"id": 1, "name": "one"}],
                "pk": "id",
            },
            headers=headers,
        )
        # Confirm drop
        response = await ds_write.client.post(
            "/data/to_drop/-/drop",
            json={"confirm": True},
            headers=headers,
        )
        assert response.json() == {"ok": True}
        assert (await ds_write.client.get("/data/to_drop")).status_code == 404

    @pytest.mark.asyncio
    async def test_create_table(self, ds_write):
        headers = _write_headers(ds_write)
        response = await ds_write.client.post(
            "/data/-/create",
            json={
                "table": "new_table",
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "title", "type": "text"},
                ],
                "pk": "id",
            },
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["ok"] is True
        assert data["table"] == "new_table"
