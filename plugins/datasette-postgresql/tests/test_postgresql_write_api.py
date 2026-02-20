"""Tests for the PostgreSQL write API.

Requires a PostgreSQL server. Set DATASETTE_TEST_POSTGRESQL=postgresql://...
to enable these tests.
"""

import os
import secrets

import pytest
import pytest_asyncio

from datasette.app import Datasette
from datasette.database import Database
from datasette_postgresql.backend import PostgresBackend


POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")

requires_postgresql = pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL=postgresql://... to run PostgreSQL tests",
)


def write_token(ds, actor_id="root", permissions=None):
    token = ds.create_token(actor_id)
    return token


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def last_event(ds):
    return ds._tracked_events[-1]


@pytest.fixture
def ds_write_pg():
    """Create a Datasette with a PostgreSQL-backed mutable database."""
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
    yield ds

    backend.close_all()
    cleanup_conn = psycopg.connect(POSTGRESQL_TEST_URL, autocommit=True)
    cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cleanup_conn.close()


@requires_postgresql
class TestWriteAPIPostgreSQL:
    @pytest.mark.asyncio
    async def test_insert_row(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Test", "score": 1.2, "age": 5}},
            headers=_headers(token),
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        rows = response.json()["rows"]
        assert len(rows) == 1
        assert rows[0]["title"] == "Test"
        assert rows[0]["score"] == 1.2
        assert rows[0]["age"] == 5
        # Verify in database
        db_rows = (
            await ds_write_pg.get_database("data").execute("select * from docs")
        ).dicts()
        assert len(db_rows) == 1
        assert db_rows[0]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_insert_rows_bulk(self, ds_write_pg):
        token = write_token(ds_write_pg)
        data = {
            "rows": [
                {"title": f"Test {i}", "score": 1.0, "age": 5} for i in range(20)
            ],
            "return": True,
        }
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json=data,
            headers=_headers(token),
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        assert len(response.json()["rows"]) == 20
        # Verify count in database
        count = (
            await ds_write_pg.get_database("data").execute(
                "select count(*) from docs"
            )
        ).rows[0][0]
        assert count == 20

    @pytest.mark.asyncio
    async def test_insert_row_alter(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={
                "row": {"title": "Test", "score": 1.2, "age": 5, "extra": "extra"},
                "alter": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True
        assert response.json()["rows"][0]["extra"] == "extra"
        # Analytics event
        event = last_event(ds_write_pg)
        assert event.name == "alter-table"
        assert "extra" in event.after_schema

    @pytest.mark.asyncio
    async def test_insert_ignore(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Insert first row
        await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"id": 1, "title": "Original"}},
            headers=_headers(token),
        )
        # Insert with ignore (should not overwrite)
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"rows": [{"id": 1, "title": "Duplicate"}], "ignore": True},
            headers=_headers(token),
        )
        assert response.status_code == 201
        row = (
            await ds_write_pg.get_database("data").execute(
                "select title from docs where id = 1"
            )
        ).rows[0][0]
        assert row == "Original"

    @pytest.mark.asyncio
    async def test_insert_replace(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Insert first row
        await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"id": 1, "title": "Original"}},
            headers=_headers(token),
        )
        # Insert with replace (should overwrite)
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"rows": [{"id": 1, "title": "Replaced"}], "replace": True},
            headers=_headers(token),
        )
        assert response.status_code == 201
        row = (
            await ds_write_pg.get_database("data").execute(
                "select title from docs where id = 1"
            )
        ).rows[0][0]
        assert row == "Replaced"

    @pytest.mark.asyncio
    async def test_upsert(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Create a table with data via the create endpoint
        create_response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "upsert_test",
                "rows": [{"id": 1, "title": "One"}],
                "pk": "id",
            },
            headers=_headers(token),
        )
        assert create_response.status_code == 201
        # Upsert: update existing + insert new
        response = await ds_write_pg.client.post(
            "/data/upsert_test/-/upsert",
            json={
                "rows": [
                    {"id": 1, "title": "Updated One"},
                    {"id": 2, "title": "Two"},
                ],
            },
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        # Verify data
        actual_rows = (
            await ds_write_pg.client.get(
                "/data/upsert_test.json?_shape=array&_sort=id"
            )
        ).json()
        assert actual_rows == [
            {"id": 1, "title": "Updated One"},
            {"id": 2, "title": "Two"},
        ]

    @pytest.mark.asyncio
    async def test_upsert_with_return(self, ds_write_pg):
        token = write_token(ds_write_pg)
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "upsert_return_test",
                "rows": [{"id": 1, "title": "One"}],
                "pk": "id",
            },
            headers=_headers(token),
        )
        response = await ds_write_pg.client.post(
            "/data/upsert_return_test/-/upsert",
            json={
                "rows": [{"id": 1, "title": "Updated"}],
                "return": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["rows"] == [{"id": 1, "title": "Updated"}]

    @pytest.mark.asyncio
    async def test_upsert_with_alter(self, ds_write_pg):
        token = write_token(ds_write_pg)
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "upsert_alter_test",
                "rows": [{"id": 1, "title": "One"}],
                "pk": "id",
            },
            headers=_headers(token),
        )
        response = await ds_write_pg.client.post(
            "/data/upsert_alter_test/-/upsert",
            json={
                "rows": [{"id": 1, "title": "Two", "extra": "extra"}],
                "alter": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 200
        actual = (
            await ds_write_pg.client.get(
                "/data/upsert_alter_test.json?_shape=array"
            )
        ).json()
        assert actual == [{"id": 1, "title": "Two", "extra": "extra"}]

    @pytest.mark.asyncio
    async def test_delete_row(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Insert a row first
        await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "To Delete", "score": 1.0, "age": 5}},
            headers=_headers(token),
        )
        # Get the ID
        rows = (
            await ds_write_pg.get_database("data").execute(
                "select id from docs"
            )
        ).rows
        pk = rows[0][0]
        # Delete it
        response = await ds_write_pg.client.post(
            f"/data/docs/{pk}/-/delete",
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        # Verify deleted
        count = (
            await ds_write_pg.get_database("data").execute(
                "select count(*) from docs"
            )
        ).rows[0][0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_compound_pk(self, ds_write_pg):
        token = write_token(ds_write_pg)
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "compound_delete",
                "rows": [{"type": "article", "key": "k", "value": "v"}],
                "pks": ["type", "key"],
            },
            headers=_headers(token),
        )
        response = await ds_write_pg.client.post(
            "/data/compound_delete/article,k/-/delete",
            headers=_headers(token),
        )
        assert response.status_code == 200
        count = (
            await ds_write_pg.get_database("data").execute(
                "select count(*) from compound_delete"
            )
        ).rows[0][0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_row(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Insert a row
        insert_response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Original", "score": 1.0, "age": 5}},
            headers=_headers(token),
        )
        pk = insert_response.json()["rows"][0]["id"]
        # Update it
        response = await ds_write_pg.client.post(
            f"/data/docs/{pk}/-/update",
            json={"update": {"title": "Updated", "score": 2.5}, "return": True},
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["row"]["title"] == "Updated"
        assert response.json()["row"]["score"] == 2.5
        assert response.json()["row"]["age"] == 5

    @pytest.mark.asyncio
    async def test_update_row_alter(self, ds_write_pg):
        token = write_token(ds_write_pg, permissions=["ur", "at"])
        # Insert a row
        insert_response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Original", "score": 1.0, "age": 5}},
            headers=_headers(write_token(ds_write_pg)),
        )
        pk = insert_response.json()["rows"][0]["id"]
        # Update with alter
        response = await ds_write_pg.client.post(
            f"/data/docs/{pk}/-/update",
            json={
                "update": {"title": "New", "extra_col": "extra_val"},
                "alter": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_update_row_null(self, ds_write_pg):
        token = write_token(ds_write_pg)
        insert_response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"row": {"title": "Original", "score": 1.0, "age": 5}},
            headers=_headers(token),
        )
        pk = insert_response.json()["rows"][0]["id"]
        response = await ds_write_pg.client.post(
            f"/data/docs/{pk}/-/update",
            json={"update": {"title": None}, "return": True},
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["row"]["title"] is None

    @pytest.mark.asyncio
    async def test_drop_table(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Create a table to drop
        await ds_write_pg.client.post(
            "/data/-/create",
            json={"table": "to_drop", "rows": [{"id": 1, "name": "one"}], "pk": "id"},
            headers=_headers(token),
        )
        # First request without confirm shows info
        response = await ds_write_pg.client.post(
            "/data/to_drop/-/drop",
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json()["row_count"] == 1
        assert response.json()["message"] == 'Pass "confirm": true to confirm'
        # Now confirm
        response2 = await ds_write_pg.client.post(
            "/data/to_drop/-/drop",
            json={"confirm": True},
            headers=_headers(token),
        )
        assert response2.json() == {"ok": True}
        # Table should be gone
        assert (
            await ds_write_pg.client.get("/data/to_drop")
        ).status_code == 404

    @pytest.mark.asyncio
    async def test_create_table_with_columns(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "new_table",
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "title", "type": "text"},
                    {"name": "score", "type": "float"},
                ],
                "pk": "id",
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        data = response.json()
        assert data["ok"] is True
        assert data["table"] == "new_table"
        assert "schema" in data

    @pytest.mark.asyncio
    async def test_create_table_with_rows(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "from_rows",
                "rows": [
                    {"id": 1, "title": "Row 1", "score": 1.5},
                    {"id": 2, "title": "Row 2", "score": 2.5},
                ],
                "pk": "id",
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        data = response.json()
        assert data["ok"] is True
        assert data["row_count"] == 2
        # Verify data
        actual = (
            await ds_write_pg.client.get(
                "/data/from_rows.json?_shape=array&_sort=id"
            )
        ).json()
        assert actual == [
            {"id": 1, "title": "Row 1", "score": 1.5},
            {"id": 2, "title": "Row 2", "score": 2.5},
        ]

    @pytest.mark.asyncio
    async def test_create_table_with_compound_pk(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "compound_pk",
                "rows": [
                    {"type": "article", "key": 123, "title": "Article 1"},
                ],
                "pks": ["type", "key"],
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        assert response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_create_table_ignore_replace(self, ds_write_pg):
        token = write_token(ds_write_pg)
        # Create table with initial data
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "rows": [
                    {"id": 1, "name": "Row 1"},
                    {"id": 2, "name": "Row 2"},
                ],
                "table": "ir_test",
                "pk": "id",
            },
            headers=_headers(token),
        )
        # Ignore duplicate
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "rows": [
                    {"id": 1, "name": "Row 1 new"},
                    {"id": 3, "name": "Row 3"},
                ],
                "table": "ir_test",
                "pk": "id",
                "ignore": True,
            },
            headers=_headers(token),
        )
        rows = (
            await ds_write_pg.client.get(
                "/data/ir_test.json?_shape=array&_sort=id"
            )
        ).json()
        assert rows == [
            {"id": 1, "name": "Row 1"},
            {"id": 2, "name": "Row 2"},
            {"id": 3, "name": "Row 3"},
        ]
        # Replace duplicate
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "rows": [{"id": 1, "name": "Row 1 replaced"}],
                "table": "ir_test",
                "pk": "id",
                "replace": True,
            },
            headers=_headers(token),
        )
        row1 = (
            await ds_write_pg.get_database("data").execute(
                "select name from ir_test where id = 1"
            )
        ).rows[0][0]
        assert row1 == "Row 1 replaced"

    @pytest.mark.asyncio
    async def test_create_table_error_duplicate(self, ds_write_pg):
        token = write_token(ds_write_pg)
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "rows": [{"id": 1, "name": "Row 1"}],
                "table": "dup_test",
                "pk": "id",
            },
            headers=_headers(token),
        )
        response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "rows": [{"id": 1, "name": "Dup"}],
                "table": "dup_test",
                "pk": "id",
            },
            headers=_headers(token),
        )
        assert response.status_code == 400
        assert response.json()["ok"] is False
        # PostgreSQL error message differs from SQLite
        assert "duplicate key" in response.json()["errors"][0].lower() or \
            "unique" in response.json()["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_create_then_alter(self, ds_write_pg):
        """Create table then add rows with new columns using alter."""
        token = write_token(ds_write_pg)
        await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "alter_test",
                "rows": [{"name": "Row 1"}],
                "pk": "id",
            },
            headers=_headers(token),
        )
        response = await ds_write_pg.client.post(
            "/data/-/create",
            json={
                "table": "alter_test",
                "rows": [{"name": "Row 2", "extra": "extra"}],
                "pk": "id",
                "alter": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        actual = (
            await ds_write_pg.client.get(
                "/data/alter_test.json?_shape=array&_sort=id"
            )
        ).json()
        assert len(actual) == 2
        assert actual[1]["extra"] == "extra"

    @pytest.mark.asyncio
    async def test_permissions_insert_denied(self, ds_write_pg):
        token = write_token(ds_write_pg, actor_id="not-root")
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={"rows": [{"title": "Test"}]},
            headers=_headers(token),
        )
        assert response.status_code == 403
        assert response.json()["errors"] == ["Permission denied"]

    @pytest.mark.asyncio
    async def test_upsert_missing_pk_error(self, ds_write_pg):
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/docs/-/upsert",
            json={"rows": [{"title": "Missing PK"}]},
            headers=_headers(token),
        )
        assert response.status_code == 400
        assert 'missing primary key' in response.json()["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_insert_return_rows(self, ds_write_pg):
        """Test that return=True returns the inserted rows with auto-generated PKs."""
        token = write_token(ds_write_pg)
        response = await ds_write_pg.client.post(
            "/data/docs/-/insert",
            json={
                "rows": [
                    {"title": "A", "score": 1.0, "age": 1},
                    {"title": "B", "score": 2.0, "age": 2},
                ],
                "return": True,
            },
            headers=_headers(token),
        )
        assert response.status_code == 201
        rows = response.json()["rows"]
        assert len(rows) == 2
        assert rows[0]["title"] == "A"
        assert rows[1]["title"] == "B"
        # Auto-generated PKs should be present
        assert "id" in rows[0]
        assert "id" in rows[1]

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, ds_write_pg):
        for path in ["/data/-/create", "/data/docs/-/drop", "/data/docs/-/insert"]:
            response = await ds_write_pg.client.get(
                path, headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 405
