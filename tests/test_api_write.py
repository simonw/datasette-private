from datasette.app import Datasette
from datasette.database import Database
from datasette.utils import sqlite3
from .utils import last_event
import os
import pytest
import secrets
import time


POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")

requires_postgresql = pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL=postgresql://... to run PostgreSQL tests",
)


@pytest.fixture
def ds_write(tmp_path_factory):
    db_directory = tmp_path_factory.mktemp("dbs")
    db_path = str(db_directory / "data.db")
    db_path_immutable = str(db_directory / "immutable.db")
    db1 = sqlite3.connect(str(db_path))
    db2 = sqlite3.connect(str(db_path_immutable))
    for db in (db1, db2):
        db.execute("vacuum")
        db.execute(
            "create table docs (id integer primary key, title text, score float, age integer)"
        )
    ds = Datasette([db_path], immutables=[db_path_immutable])
    ds.root_enabled = True
    yield ds
    # Close both setup connections plus any Datasette-managed connections.
    db1.close()
    db2.close()
    for database in ds.databases.values():
        if not database.is_memory:
            database.close()


def write_token(ds, actor_id="root", permissions=None):
    to_sign = {"a": actor_id, "token": "dstok", "t": int(time.time())}
    if permissions:
        to_sign["_r"] = {"a": permissions}
    return "dstok_{}".format(ds.sign(to_sign, namespace="token"))


def _headers(token):
    return {
        "Authorization": "Bearer {}".format(token),
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    (
        "application/json",
        "application/json; charset=utf-8",
    ),
)
async def test_insert_row(ds_write, content_type):
    token = write_token(ds_write)
    response = await ds_write.client.post(
        "/data/docs/-/insert",
        json={"row": {"title": "Test", "score": 1.2, "age": 5}},
        headers={
            "Authorization": "Bearer {}".format(token),
            "Content-Type": content_type,
        },
    )
    expected_row = {"id": 1, "title": "Test", "score": 1.2, "age": 5}
    assert response.status_code == 201
    assert response.json()["ok"] is True
    assert response.json()["rows"] == [expected_row]
    rows = (await ds_write.get_database("data").execute("select * from docs")).dicts()
    assert rows[0] == expected_row
    # Analytics event
    event = last_event(ds_write)
    assert event.name == "insert-rows"
    assert event.num_rows == 1
    assert event.database == "data"
    assert event.table == "docs"
    assert not event.ignore
    assert not event.replace


@pytest.mark.asyncio
async def test_insert_row_alter(ds_write):
    token = write_token(ds_write)
    response = await ds_write.client.post(
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
    event = last_event(ds_write)
    assert event.name == "alter-table"
    assert "extra" not in event.before_schema
    assert "extra" in event.after_schema


@pytest.mark.asyncio
@pytest.mark.parametrize("return_rows", (True, False))
async def test_insert_rows(ds_write, return_rows):
    token = write_token(ds_write)
    data = {
        "rows": [
            {"title": "Test {}".format(i), "score": 1.0, "age": 5} for i in range(20)
        ]
    }
    if return_rows:
        data["return"] = True
    response = await ds_write.client.post(
        "/data/docs/-/insert",
        json=data,
        headers=_headers(token),
    )
    assert response.status_code == 201

    # Analytics event
    event = last_event(ds_write)
    assert event.name == "insert-rows"
    assert event.num_rows == 20
    assert event.database == "data"
    assert event.table == "docs"
    assert not event.ignore
    assert not event.replace

    actual_rows = (
        await ds_write.get_database("data").execute("select * from docs")
    ).dicts()
    assert len(actual_rows) == 20
    assert actual_rows == [
        {"id": i + 1, "title": "Test {}".format(i), "score": 1.0, "age": 5}
        for i in range(20)
    ]
    assert response.json()["ok"] is True
    if return_rows:
        assert response.json()["rows"] == actual_rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,input,special_case,expected_status,expected_errors",
    (
        (
            "/data2/docs/-/insert",
            {},
            None,
            404,
            ["Database not found"],
        ),
        (
            "/data/docs2/-/insert",
            {},
            None,
            404,
            ["Table not found"],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"} for i in range(10)]},
            "bad_token",
            403,
            ["Permission denied"],
        ),
        (
            "/data/docs/-/insert",
            {},
            "invalid_json",
            400,
            [
                "Invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
            ],
        ),
        (
            "/data/docs/-/insert",
            {},
            "invalid_content_type",
            400,
            ["Invalid content-type, must be application/json"],
        ),
        (
            "/data/docs/-/insert",
            [],
            None,
            400,
            ["JSON must be a dictionary"],
        ),
        (
            "/data/docs/-/insert",
            {"row": "blah"},
            None,
            400,
            ['"row" must be a dictionary'],
        ),
        (
            "/data/docs/-/insert",
            {"blah": "blah"},
            None,
            400,
            ['JSON must have one or other of "row" or "rows"'],
        ),
        (
            "/data/docs/-/insert",
            {"rows": "blah"},
            None,
            400,
            ['"rows" must be a list'],
        ),
        (
            "/data/docs/-/insert",
            {"rows": ["blah"]},
            None,
            400,
            ['"rows" must be a list of dictionaries'],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"} for i in range(101)]},
            None,
            400,
            ["Too many rows, maximum allowed is 100"],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"id": 1, "title": "Test"}, {"id": 2, "title": "Test"}]},
            "duplicate_id",
            400,
            ["UNIQUE constraint failed: docs.id"],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"}], "ignore": True, "replace": True},
            None,
            400,
            ['Cannot use "ignore" and "replace" at the same time'],
        ),
        (
            # Replace is not allowed if you don't have update-row
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"}], "replace": True},
            "insert-but-not-update",
            403,
            ['Permission denied: need update-row to use "replace"'],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"}], "invalid_param": True},
            None,
            400,
            ['Invalid parameter: "invalid_param"'],
        ),
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test"}], "one": True, "two": True},
            None,
            400,
            ['Invalid parameter: "one", "two"'],
        ),
        (
            "/immutable/docs/-/insert",
            {"rows": [{"title": "Test"}]},
            None,
            403,
            ["Database is immutable"],
        ),
        # Validate columns of each row
        (
            "/data/docs/-/insert",
            {"rows": [{"title": "Test", "bad": 1, "worse": 2} for i in range(2)]},
            None,
            400,
            [
                "Row 0 has invalid columns: bad, worse",
                "Row 1 has invalid columns: bad, worse",
            ],
        ),
        ## UPSERT ERRORS:
        (
            "/immutable/docs/-/upsert",
            {"rows": [{"title": "Test"}]},
            None,
            403,
            ["Database is immutable"],
        ),
        (
            "/data/badtable/-/upsert",
            {"rows": [{"title": "Test"}]},
            None,
            404,
            ["Table not found"],
        ),
        # missing primary key
        (
            "/data/docs/-/upsert",
            {"rows": [{"title": "Missing PK"}]},
            None,
            400,
            ['Row 0 is missing primary key column(s): "id"'],
        ),
        # Upsert does not support ignore or replace
        (
            "/data/docs/-/upsert",
            {"rows": [{"id": 1, "title": "Bad"}], "ignore": True},
            None,
            400,
            ["Upsert does not support ignore or replace"],
        ),
        # Upsert permissions
        (
            "/data/docs/-/upsert",
            {"rows": [{"id": 1, "title": "Disallowed"}]},
            "insert-but-not-update",
            403,
            ["Permission denied: need both insert-row and update-row"],
        ),
        (
            "/data/docs/-/upsert",
            {"rows": [{"id": 1, "title": "Disallowed"}]},
            "update-but-not-insert",
            403,
            ["Permission denied: need both insert-row and update-row"],
        ),
        # Alter table forbidden without alter permission
        (
            "/data/docs/-/upsert",
            {"rows": [{"id": 1, "title": "One", "extra": "extra"}], "alter": True},
            "update-and-insert-but-no-alter",
            403,
            ["Permission denied for alter-table"],
        ),
    ),
)
async def test_insert_or_upsert_row_errors(
    ds_write, path, input, special_case, expected_status, expected_errors
):
    token_permissions = []
    if special_case == "insert-but-not-update":
        token_permissions = ["ir", "vi"]
    if special_case == "update-but-not-insert":
        token_permissions = ["ur", "vi"]
    if special_case == "update-and-insert-but-no-alter":
        token_permissions = ["ur", "ir"]
    token = write_token(ds_write, permissions=token_permissions)
    if special_case == "duplicate_id":
        await ds_write.get_database("data").execute_write(
            "insert into docs (id) values (1)"
        )
    if special_case == "bad_token":
        token += "bad"
    kwargs = dict(
        json=input,
        headers={
            "Authorization": "Bearer {}".format(token),
            "Content-Type": (
                "text/plain"
                if special_case == "invalid_content_type"
                else "application/json"
            ),
        },
    )

    actor_response = (
        await ds_write.client.get("/-/actor.json", headers=kwargs["headers"])
    ).json()
    assert set((actor_response["actor"] or {}).get("_r", {}).get("a") or []) == set(
        token_permissions
    )

    if special_case == "invalid_json":
        del kwargs["json"]
        kwargs["content"] = "{bad json"
    before_count = (
        await ds_write.get_database("data").execute("select count(*) from docs")
    ).rows[0][0] == 0
    response = await ds_write.client.post(
        path,
        **kwargs,
    )
    assert response.status_code == expected_status
    assert response.json()["ok"] is False
    assert response.json()["errors"] == expected_errors
    # Check that no rows were inserted
    after_count = (
        await ds_write.get_database("data").execute("select count(*) from docs")
    ).rows[0][0] == 0
    assert before_count == after_count


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", (True, False))
async def test_upsert_permissions_per_table(ds_write, allowed):
    # https://github.com/simonw/datasette/issues/2262
    token = "dstok_{}".format(
        ds_write.sign(
            {
                "a": "root",
                "token": "dstok",
                "t": int(time.time()),
                "_r": {
                    "r": {
                        "data": {
                            "docs" if allowed else "other": ["ir", "ur"],
                        }
                    }
                },
            },
            namespace="token",
        )
    )
    response = await ds_write.client.post(
        "/data/docs/-/upsert",
        json={"rows": [{"id": 1, "title": "One"}]},
        headers={
            "Authorization": "Bearer {}".format(token),
        },
    )
    if allowed:
        assert response.status_code == 200
        assert response.json()["ok"] is True
    else:
        assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ignore,replace,expected_rows",
    (
        (
            True,
            False,
            [
                {"id": 1, "title": "Exists", "score": None, "age": None},
            ],
        ),
        (
            False,
            True,
            [
                {"id": 1, "title": "One", "score": None, "age": None},
            ],
        ),
    ),
)
@pytest.mark.parametrize("should_return", (True, False))
async def test_insert_ignore_replace(
    ds_write, ignore, replace, expected_rows, should_return
):
    await ds_write.get_database("data").execute_write(
        "insert into docs (id, title) values (1, 'Exists')"
    )
    token = write_token(ds_write)
    data = {"rows": [{"id": 1, "title": "One"}]}
    if ignore:
        data["ignore"] = True
    if replace:
        data["replace"] = True
    if should_return:
        data["return"] = True
    response = await ds_write.client.post(
        "/data/docs/-/insert",
        json=data,
        headers=_headers(token),
    )
    assert response.status_code == 201

    # Analytics event
    event = last_event(ds_write)
    assert event.name == "insert-rows"
    assert event.num_rows == 1
    assert event.database == "data"
    assert event.table == "docs"
    assert event.ignore == ignore
    assert event.replace == replace

    actual_rows = (
        await ds_write.get_database("data").execute("select * from docs")
    ).dicts()

    assert actual_rows == expected_rows
    assert response.json()["ok"] is True
    if should_return:
        assert response.json()["rows"] == expected_rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial,input,expected_rows",
    (
        (
            # Simple primary key update
            {"rows": [{"id": 1, "title": "One"}], "pk": "id"},
            {"rows": [{"id": 1, "title": "Two"}]},
            [
                {"id": 1, "title": "Two"},
            ],
        ),
        (
            # Multiple rows update one of them
            {
                "rows": [{"id": 1, "title": "One"}, {"id": 2, "title": "Two"}],
                "pk": "id",
            },
            {"rows": [{"id": 1, "title": "Three"}]},
            [
                {"id": 1, "title": "Three"},
                {"id": 2, "title": "Two"},
            ],
        ),
        (
            # rowid update
            {"rows": [{"title": "One"}]},
            {"rows": [{"rowid": 1, "title": "Two"}]},
            [
                {"rowid": 1, "title": "Two"},
            ],
        ),
        (
            # Compound primary key update
            {"rows": [{"id": 1, "title": "One", "score": 1}], "pks": ["id", "score"]},
            {"rows": [{"id": 1, "title": "Two", "score": 1}]},
            [
                {"id": 1, "title": "Two", "score": 1},
            ],
        ),
        (
            # Upsert with an alter
            {"rows": [{"id": 1, "title": "One"}], "pk": "id"},
            {"rows": [{"id": 1, "title": "Two", "extra": "extra"}], "alter": True},
            [{"id": 1, "title": "Two", "extra": "extra"}],
        ),
    ),
)
@pytest.mark.parametrize("should_return", (False, True))
async def test_upsert(ds_write, initial, input, expected_rows, should_return):
    token = write_token(ds_write)
    # Insert initial data
    initial["table"] = "upsert_test"
    create_response = await ds_write.client.post(
        "/data/-/create",
        json=initial,
        headers=_headers(token),
    )
    assert create_response.status_code == 201
    if should_return:
        input["return"] = True
    response = await ds_write.client.post(
        "/data/upsert_test/-/upsert",
        json=input,
        headers=_headers(token),
    )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True

    # Analytics event
    event = last_event(ds_write)
    assert event.database == "data"
    assert event.table == "upsert_test"
    if input.get("alter"):
        assert event.name == "alter-table"
        assert "extra" in event.after_schema
    else:
        assert event.name == "upsert-rows"
        assert event.num_rows == 1

    if should_return:
        # We only expect it to return rows corresponding to those we sent
        expected_returned_rows = expected_rows[: len(input["rows"])]
        assert response.json()["rows"] == expected_returned_rows
    # Check the database too
    actual_rows = (
        await ds_write.client.get("/data/upsert_test.json?_shape=array")
    ).json()
    assert actual_rows == expected_rows
    # Drop the upsert_test table
    await ds_write.get_database("data").execute_write("drop table upsert_test")


async def _insert_row(ds):
    insert_response = await ds.client.post(
        "/data/docs/-/insert",
        json={"row": {"title": "Row one", "score": 1.2, "age": 5}, "return": True},
        headers=_headers(write_token(ds)),
    )
    assert insert_response.status_code == 201
    return insert_response.json()["rows"][0]["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("no_token", "no_perm", "bad_table"))
async def test_delete_row_errors(ds_write, scenario):
    if scenario == "no_token":
        token = "bad_token"
    elif scenario == "no_perm":
        token = write_token(ds_write, actor_id="not-root")
    else:
        token = write_token(ds_write)

    pk = await _insert_row(ds_write)

    path = "/data/{}/{}/-/delete".format(
        "docs" if scenario != "bad_table" else "bad_table", pk
    )
    response = await ds_write.client.post(
        path,
        headers=_headers(token),
    )
    assert response.status_code == 403 if scenario in ("no_token", "bad_token") else 404
    assert response.json()["ok"] is False
    assert (
        response.json()["errors"] == ["Permission denied"]
        if scenario == "no_token"
        else ["Table not found"]
    )
    assert len((await ds_write.client.get("/data/docs.json?_shape=array")).json()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table,row_for_create,pks,delete_path",
    (
        ("rowid_table", {"name": "rowid row"}, None, None),
        ("pk_table", {"id": 1, "name": "ID table"}, "id", "1"),
        (
            "compound_pk_table",
            {"type": "article", "key": "k"},
            ["type", "key"],
            "article,k",
        ),
    ),
)
async def test_delete_row(ds_write, table, row_for_create, pks, delete_path):
    # First create the table with that example row
    create_data = {
        "table": table,
        "row": row_for_create,
    }
    if pks:
        if isinstance(pks, str):
            create_data["pk"] = pks
        else:
            create_data["pks"] = pks
    create_response = await ds_write.client.post(
        "/data/-/create",
        json=create_data,
        headers=_headers(write_token(ds_write)),
    )
    assert create_response.status_code == 201, create_response.json()
    # Should be a single row
    assert (
        await ds_write.client.get(
            "/data/-/query.json?_shape=arrayfirst&sql=select+count(*)+from+{}".format(
                table
            )
        )
    ).json() == [1]
    # Now delete the row
    if delete_path is None:
        # Special case for that rowid table
        delete_path = (
            await ds_write.client.get(
                "/data/-/query.json?_shape=arrayfirst&sql=select+rowid+from+{}".format(
                    table
                )
            )
        ).json()[0]

    delete_response = await ds_write.client.post(
        "/data/{}/{}/-/delete".format(table, delete_path),
        headers=_headers(write_token(ds_write)),
    )
    assert delete_response.status_code == 200

    # Analytics event
    event = last_event(ds_write)
    assert event.name == "delete-row"
    assert event.database == "data"
    assert event.table == table
    assert event.pks == str(delete_path).split(",")
    assert (
        await ds_write.client.get(
            "/data/-/query.json?_shape=arrayfirst&sql=select+count(*)+from+{}".format(
                table
            )
        )
    ).json() == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ("no_token", "no_perm", "bad_table", "cannot_alter")
)
async def test_update_row_check_permission(ds_write, scenario):
    if scenario == "no_token":
        token = "bad_token"
    elif scenario == "no_perm":
        token = write_token(ds_write, actor_id="not-root")
    elif scenario == "cannot_alter":
        # update-row but no alter-table:
        token = write_token(ds_write, permissions=["ur"])
    else:
        token = write_token(ds_write)

    pk = await _insert_row(ds_write)

    path = "/data/{}/{}/-/update".format(
        "docs" if scenario != "bad_table" else "bad_table", pk
    )

    json_body = {"update": {"title": "New title"}}
    if scenario == "cannot_alter":
        json_body["alter"] = True

    response = await ds_write.client.post(
        path,
        json=json_body,
        headers=_headers(token),
    )
    assert response.status_code == 403 if scenario in ("no_token", "bad_token") else 404
    assert response.json()["ok"] is False
    assert (
        response.json()["errors"] == ["Permission denied"]
        if scenario == "no_token"
        else ["Table not found"]
    )


@pytest.mark.asyncio
async def test_update_row_invalid_key(ds_write):
    token = write_token(ds_write)

    pk = await _insert_row(ds_write)

    path = "/data/docs/{}/-/update".format(pk)
    response = await ds_write.client.post(
        path,
        json={"update": {"title": "New title"}, "bad_key": 1},
        headers=_headers(token),
    )
    assert response.status_code == 400
    assert response.json() == {"ok": False, "errors": ["Invalid keys: bad_key"]}


@pytest.mark.asyncio
async def test_update_row_alter(ds_write):
    token = write_token(ds_write, permissions=["ur", "at"])
    pk = await _insert_row(ds_write)
    path = "/data/docs/{}/-/update".format(pk)
    response = await ds_write.client.post(
        path,
        json={"update": {"title": "New title", "extra": "extra"}, "alter": True},
        headers=_headers(token),
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input,expected_errors",
    (
        ({"title": "New title"}, None),
        ({"title": None}, None),
        ({"score": 1.6}, None),
        ({"age": 10}, None),
        ({"title": "New title", "score": 1.6}, None),
        ({"title2": "New title"}, ["no such column: title2"]),
    ),
)
@pytest.mark.parametrize("use_return", (True, False))
async def test_update_row(ds_write, input, expected_errors, use_return):
    token = write_token(ds_write)
    pk = await _insert_row(ds_write)

    path = "/data/docs/{}/-/update".format(pk)

    data = {"update": input}
    if use_return:
        data["return"] = True

    response = await ds_write.client.post(
        path,
        json=data,
        headers=_headers(token),
    )
    if expected_errors:
        assert response.status_code == 400
        assert response.json()["ok"] is False
        assert response.json()["errors"] == expected_errors
        return

    assert response.json()["ok"] is True
    if not use_return:
        assert "row" not in response.json()
    else:
        returned_row = response.json()["row"]
        assert returned_row["id"] == pk
        for k, v in input.items():
            assert returned_row[k] == v

    # Analytics event
    event = last_event(ds_write)
    assert event.actor == {"id": "root", "token": "dstok"}
    assert event.database == "data"
    assert event.table == "docs"
    assert event.pks == [str(pk)]

    # And fetch the row to check it's updated
    response = await ds_write.client.get(
        "/data/docs/{}.json?_shape=array".format(pk),
    )
    assert response.status_code == 200
    row = response.json()[0]
    assert row["id"] == pk
    for k, v in input.items():
        assert row[k] == v


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ("no_token", "no_perm", "bad_table", "has_perm", "immutable")
)
async def test_drop_table(ds_write, scenario):
    if scenario == "no_token":
        token = "bad_token"
    elif scenario == "no_perm":
        token = write_token(ds_write, actor_id="not-root")
    else:
        token = write_token(ds_write)
    should_work = scenario == "has_perm"
    await ds_write.get_database("data").execute_write(
        "insert into docs (id, title) values (1, 'Row 1')"
    )
    path = "/{database}/{table}/-/drop".format(
        database="immutable" if scenario == "immutable" else "data",
        table="docs" if scenario != "bad_table" else "bad_table",
    )
    response = await ds_write.client.post(
        path,
        headers=_headers(token),
    )
    if not should_work:
        assert (
            response.status_code == 403
            if scenario in ("no_token", "bad_token")
            else 404
        )
        assert response.json()["ok"] is False
        expected_error = "Permission denied"
        if scenario == "bad_table":
            expected_error = "Table not found"
        elif scenario == "immutable":
            expected_error = "Database is immutable"
        assert response.json()["errors"] == [expected_error]
        assert (await ds_write.client.get("/data/docs")).status_code == 200
    else:
        # It should show a confirmation page
        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "database": "data",
            "table": "docs",
            "row_count": 1,
            "message": 'Pass "confirm": true to confirm',
        }
        assert (await ds_write.client.get("/data/docs")).status_code == 200
        # Now send confirm: true
        response2 = await ds_write.client.post(
            path,
            json={"confirm": True},
            headers=_headers(token),
        )
        assert response2.json() == {"ok": True}
        # Check event
        event = last_event(ds_write)
        assert event.name == "drop-table"
        assert event.actor == {"id": "root", "token": "dstok"}
        assert event.table == "docs"
        assert event.database == "data"
        # Table should 404
        assert (await ds_write.client.get("/data/docs")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input,expected_status,expected_response,expected_events",
    (
        # Permission error with a bad token
        (
            {"table": "bad", "row": {"id": 1}},
            403,
            {"ok": False, "errors": ["Permission denied"]},
            [],
        ),
        # Successful creation with columns:
        (
            {
                "table": "one",
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                    },
                    {
                        "name": "title",
                        "type": "text",
                    },
                    {
                        "name": "score",
                        "type": "integer",
                    },
                    {
                        "name": "weight",
                        "type": "float",
                    },
                    {
                        "name": "thumbnail",
                        "type": "blob",
                    },
                ],
                "pk": "id",
            },
            201,
            {
                "ok": True,
                "database": "data",
                "table": "one",
                "table_url": "http://localhost/data/one",
                "table_api_url": "http://localhost/data/one.json",
                "schema": (
                    "CREATE TABLE [one] (\n"
                    "   [id] INTEGER PRIMARY KEY,\n"
                    "   [title] TEXT,\n"
                    "   [score] INTEGER,\n"
                    "   [weight] FLOAT,\n"
                    "   [thumbnail] BLOB\n"
                    ")"
                ),
            },
            ["create-table"],
        ),
        # Successful creation with rows:
        (
            {
                "table": "two",
                "rows": [
                    {
                        "id": 1,
                        "title": "Row 1",
                        "score": 1.5,
                    },
                    {
                        "id": 2,
                        "title": "Row 2",
                        "score": 1.5,
                    },
                ],
                "pk": "id",
            },
            201,
            {
                "ok": True,
                "database": "data",
                "table": "two",
                "table_url": "http://localhost/data/two",
                "table_api_url": "http://localhost/data/two.json",
                "schema": (
                    "CREATE TABLE [two] (\n"
                    "   [id] INTEGER PRIMARY KEY,\n"
                    "   [title] TEXT,\n"
                    "   [score] FLOAT\n"
                    ")"
                ),
                "row_count": 2,
            },
            ["create-table", "insert-rows"],
        ),
        # Successful creation with row:
        (
            {
                "table": "three",
                "row": {
                    "id": 1,
                    "title": "Row 1",
                    "score": 1.5,
                },
                "pk": "id",
            },
            201,
            {
                "ok": True,
                "database": "data",
                "table": "three",
                "table_url": "http://localhost/data/three",
                "table_api_url": "http://localhost/data/three.json",
                "schema": (
                    "CREATE TABLE [three] (\n"
                    "   [id] INTEGER PRIMARY KEY,\n"
                    "   [title] TEXT,\n"
                    "   [score] FLOAT\n"
                    ")"
                ),
                "row_count": 1,
            },
            ["create-table", "insert-rows"],
        ),
        # Create with row and no primary key
        (
            {
                "table": "four",
                "row": {
                    "name": "Row 1",
                },
            },
            201,
            {
                "ok": True,
                "database": "data",
                "table": "four",
                "table_url": "http://localhost/data/four",
                "table_api_url": "http://localhost/data/four.json",
                "schema": ("CREATE TABLE [four] (\n" "   [name] TEXT\n" ")"),
                "row_count": 1,
            },
            ["create-table", "insert-rows"],
        ),
        # Create table with compound primary key
        (
            {
                "table": "five",
                "row": {"type": "article", "key": 123, "title": "Article 1"},
                "pks": ["type", "key"],
            },
            201,
            {
                "ok": True,
                "database": "data",
                "table": "five",
                "table_url": "http://localhost/data/five",
                "table_api_url": "http://localhost/data/five.json",
                "schema": (
                    "CREATE TABLE [five] (\n   [type] TEXT,\n   [key] INTEGER,\n"
                    "   [title] TEXT,\n   PRIMARY KEY ([type], [key])\n)"
                ),
                "row_count": 1,
            },
            ["create-table", "insert-rows"],
        ),
        # Error: Table is required
        (
            {
                "row": {"id": 1},
            },
            400,
            {
                "ok": False,
                "errors": ["Table is required"],
            },
            [],
        ),
        # Error: Invalid table name
        (
            {
                "table": "sqlite_bad_name",
                "row": {"id": 1},
            },
            400,
            {
                "ok": False,
                "errors": ["Invalid table name"],
            },
            [],
        ),
        # Error: JSON must be an object
        (
            [],
            400,
            {
                "ok": False,
                "errors": ["JSON must be an object"],
            },
            [],
        ),
        # Error: Cannot specify columns with rows or row
        (
            {
                "table": "bad",
                "columns": [{"name": "id", "type": "integer"}],
                "rows": [{"id": 1}],
            },
            400,
            {
                "ok": False,
                "errors": ["Cannot specify columns with rows or row"],
            },
            [],
        ),
        # Error: columns, rows or row is required
        (
            {
                "table": "bad",
            },
            400,
            {
                "ok": False,
                "errors": ["columns, rows or row is required"],
            },
            [],
        ),
        # Error: columns must be a list
        (
            {
                "table": "bad",
                "columns": {"name": "id", "type": "integer"},
            },
            400,
            {
                "ok": False,
                "errors": ["columns must be a list"],
            },
            [],
        ),
        # Error: columns must be a list of objects
        (
            {
                "table": "bad",
                "columns": ["id"],
            },
            400,
            {
                "ok": False,
                "errors": ["columns must be a list of objects"],
            },
            [],
        ),
        # Error: Column name is required
        (
            {
                "table": "bad",
                "columns": [{"type": "integer"}],
            },
            400,
            {
                "ok": False,
                "errors": ["Column name is required"],
            },
            [],
        ),
        # Error: Unsupported column type
        (
            {
                "table": "bad",
                "columns": [{"name": "id", "type": "bad"}],
            },
            400,
            {
                "ok": False,
                "errors": ["Unsupported column type: bad"],
            },
            [],
        ),
        # Error: Duplicate column name
        (
            {
                "table": "bad",
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "id", "type": "integer"},
                ],
            },
            400,
            {
                "ok": False,
                "errors": ["Duplicate column name: id"],
            },
            [],
        ),
        # Error: rows must be a list
        (
            {
                "table": "bad",
                "rows": {"id": 1},
            },
            400,
            {
                "ok": False,
                "errors": ["rows must be a list"],
            },
            [],
        ),
        # Error: rows must be a list of objects
        (
            {
                "table": "bad",
                "rows": ["id"],
            },
            400,
            {
                "ok": False,
                "errors": ["rows must be a list of objects"],
            },
            [],
        ),
        # Error: pk must be a string
        (
            {
                "table": "bad",
                "row": {"id": 1},
                "pk": 1,
            },
            400,
            {
                "ok": False,
                "errors": ["pk must be a string"],
            },
            [],
        ),
        # Error: Cannot specify both pk and pks
        (
            {
                "table": "bad",
                "row": {"id": 1, "name": "Row 1"},
                "pk": "id",
                "pks": ["id", "name"],
            },
            400,
            {
                "ok": False,
                "errors": ["Cannot specify both pk and pks"],
            },
            [],
        ),
        # Error: pks must be a list
        (
            {
                "table": "bad",
                "row": {"id": 1, "name": "Row 1"},
                "pks": "id",
            },
            400,
            {
                "ok": False,
                "errors": ["pks must be a list"],
            },
            [],
        ),
        # Error: pks must be a list of strings
        (
            {"table": "bad", "row": {"id": 1, "name": "Row 1"}, "pks": [1, 2]},
            400,
            {"ok": False, "errors": ["pks must be a list of strings"]},
            [],
        ),
        # Error: ignore and replace are mutually exclusive
        (
            {
                "table": "bad",
                "row": {"id": 1, "name": "Row 1"},
                "pk": "id",
                "ignore": True,
                "replace": True,
            },
            400,
            {
                "ok": False,
                "errors": ["ignore and replace are mutually exclusive"],
            },
            [],
        ),
        # ignore and replace require row or rows
        (
            {
                "table": "bad",
                "columns": [{"name": "id", "type": "integer"}],
                "ignore": True,
            },
            400,
            {
                "ok": False,
                "errors": ["ignore and replace require row or rows"],
            },
            [],
        ),
        # ignore and replace require pk or pks
        (
            {
                "table": "bad",
                "row": {"id": 1},
                "ignore": True,
            },
            400,
            {
                "ok": False,
                "errors": ["ignore and replace require pk or pks"],
            },
            [],
        ),
        (
            {
                "table": "bad",
                "row": {"id": 1},
                "replace": True,
            },
            400,
            {
                "ok": False,
                "errors": ["ignore and replace require pk or pks"],
            },
            [],
        ),
    ),
)
async def test_create_table(
    ds_write, input, expected_status, expected_response, expected_events
):
    ds_write._tracked_events = []
    # Special case for expected status of 403
    if expected_status == 403:
        token = "bad_token"
    else:
        token = write_token(ds_write)
    response = await ds_write.client.post(
        "/data/-/create",
        json=input,
        headers=_headers(token),
    )
    assert response.status_code == expected_status
    data = response.json()
    assert data == expected_response
    # Should have tracked the expected events
    events = ds_write._tracked_events
    assert [e.name for e in events] == expected_events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "permissions,body,expected_status,expected_errors",
    (
        (["create-table"], {"table": "t", "columns": [{"name": "c"}]}, 201, None),
        # Need insert-row too if you use "rows":
        (
            ["create-table"],
            {"table": "t", "rows": [{"name": "c"}]},
            403,
            ["Permission denied: need insert-row"],
        ),
        # This should work:
        (
            ["create-table", "insert-row"],
            {"table": "t", "rows": [{"name": "c"}]},
            201,
            None,
        ),
        # If you use replace: true you need update-row too:
        (
            ["create-table", "insert-row"],
            {"table": "t", "rows": [{"id": 1}], "pk": "id", "replace": True},
            403,
            ["Permission denied: need update-row"],
        ),
    ),
)
async def test_create_table_permissions(
    ds_write, permissions, body, expected_status, expected_errors
):
    token = ds_write.create_token("root", restrict_all=["view-instance"] + permissions)
    response = await ds_write.client.post(
        "/data/-/create",
        json=body,
        headers=_headers(token),
    )
    assert response.status_code == expected_status
    if expected_errors:
        data = response.json()
        assert data["ok"] is False
        assert data["errors"] == expected_errors


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Flaky, see https://github.com/simonw/datasette/issues/2356")
@pytest.mark.parametrize(
    "input,expected_rows_after",
    (
        (
            {
                "table": "test_insert_replace",
                "rows": [
                    {"id": 1, "name": "Row 1 new"},
                    {"id": 3, "name": "Row 3 new"},
                ],
                "pk": "id",
                "ignore": True,
            },
            [
                {"id": 1, "name": "Row 1"},
                {"id": 2, "name": "Row 2"},
                {"id": 3, "name": "Row 3 new"},
            ],
        ),
        (
            {
                "table": "test_insert_replace",
                "rows": [
                    {"id": 1, "name": "Row 1 new"},
                    {"id": 3, "name": "Row 3 new"},
                ],
                "pk": "id",
                "replace": True,
            },
            [
                {"id": 1, "name": "Row 1 new"},
                {"id": 2, "name": "Row 2"},
                {"id": 3, "name": "Row 3 new"},
            ],
        ),
    ),
)
async def test_create_table_ignore_replace(ds_write, input, expected_rows_after):
    # Create table with two rows
    token = write_token(ds_write)
    first_response = await ds_write.client.post(
        "/data/-/create",
        json={
            "rows": [{"id": 1, "name": "Row 1"}, {"id": 2, "name": "Row 2"}],
            "table": "test_insert_replace",
            "pk": "id",
        },
        headers=_headers(token),
    )
    assert first_response.status_code == 201

    ds_write._tracked_events = []

    # Try a second time
    second_response = await ds_write.client.post(
        "/data/-/create",
        json=input,
        headers=_headers(token),
    )
    assert second_response.status_code == 201
    # Check that the rows are as expected
    rows = await ds_write.client.get("/data/test_insert_replace.json?_shape=array")
    assert rows.json() == expected_rows_after

    # Check it fired the right events
    event_names = [e.name for e in ds_write._tracked_events]
    assert event_names == ["insert-rows"]


@pytest.mark.asyncio
async def test_create_table_error_if_pk_changed(ds_write):
    token = write_token(ds_write)
    first_response = await ds_write.client.post(
        "/data/-/create",
        json={
            "rows": [{"id": 1, "name": "Row 1"}, {"id": 2, "name": "Row 2"}],
            "table": "test_insert_replace",
            "pk": "id",
        },
        headers=_headers(token),
    )
    assert first_response.status_code == 201
    # Try a second time with a different pk
    second_response = await ds_write.client.post(
        "/data/-/create",
        json={
            "rows": [{"id": 1, "name": "Row 1"}, {"id": 2, "name": "Row 2"}],
            "table": "test_insert_replace",
            "pk": "name",
            "replace": True,
        },
        headers=_headers(token),
    )
    assert second_response.status_code == 400
    assert second_response.json() == {
        "ok": False,
        "errors": ["pk cannot be changed for existing table"],
    }


@pytest.mark.asyncio
async def test_create_table_error_rows_twice_with_duplicates(ds_write):
    # Error if you don't send ignore: True or replace: True
    token = write_token(ds_write)
    input = {
        "rows": [{"id": 1, "name": "Row 1"}, {"id": 2, "name": "Row 2"}],
        "table": "test_create_twice",
        "pk": "id",
    }
    first_response = await ds_write.client.post(
        "/data/-/create",
        json=input,
        headers=_headers(token),
    )
    assert first_response.status_code == 201
    second_response = await ds_write.client.post(
        "/data/-/create",
        json=input,
        headers=_headers(token),
    )
    assert second_response.status_code == 400
    assert second_response.json() == {
        "ok": False,
        "errors": ["UNIQUE constraint failed: test_create_twice.id"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    (
        "/data/-/create",
        "/data/docs/-/drop",
        "/data/docs/-/insert",
    ),
)
async def test_method_not_allowed(ds_write, path):
    response = await ds_write.client.get(
        path,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 405
    assert response.json() == {
        "ok": False,
        "error": "Method not allowed",
    }


@pytest.mark.asyncio
async def test_create_uses_alter_by_default_for_new_table(ds_write):
    ds_write._tracked_events = []
    token = write_token(ds_write)
    response = await ds_write.client.post(
        "/data/-/create",
        json={
            "table": "new_table",
            "rows": [
                {
                    "name": "Row 1",
                }
            ]
            * 100
            + [
                {"name": "Row 2", "extra": "Extra"},
            ],
            "pk": "id",
        },
        headers=_headers(token),
    )
    assert response.status_code == 201
    event_names = [e.name for e in ds_write._tracked_events]
    assert event_names == ["create-table", "insert-rows"]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_alter_permission", (True, False))
async def test_create_using_alter_against_existing_table(
    ds_write, has_alter_permission
):
    token = write_token(
        ds_write, permissions=["ir", "ct"] + (["at"] if has_alter_permission else [])
    )
    # First create the table
    response = await ds_write.client.post(
        "/data/-/create",
        json={
            "table": "new_table",
            "rows": [
                {
                    "name": "Row 1",
                }
            ],
            "pk": "id",
        },
        headers=_headers(token),
    )
    assert response.status_code == 201

    ds_write._tracked_events = []
    # Now try to insert more rows using /-/create with alter=True
    response2 = await ds_write.client.post(
        "/data/-/create",
        json={
            "table": "new_table",
            "rows": [{"name": "Row 2", "extra": "extra"}],
            "pk": "id",
            "alter": True,
        },
        headers=_headers(token),
    )
    if not has_alter_permission:
        assert response2.status_code == 403
        assert response2.json() == {
            "ok": False,
            "errors": ["Permission denied: need alter-table"],
        }
    else:
        assert response2.status_code == 201

        event_names = [e.name for e in ds_write._tracked_events]
        assert event_names == ["alter-table", "insert-rows"]

        # It should have altered the table
        alter_event = ds_write._tracked_events[0]
        assert alter_event.name == "alter-table"
        assert "extra" not in alter_event.before_schema
        assert "extra" in alter_event.after_schema

        insert_rows_event = ds_write._tracked_events[1]
        assert insert_rows_event.name == "insert-rows"
        assert insert_rows_event.num_rows == 1


# ---- PostgreSQL write API tests ----


@pytest.fixture
def ds_write_pg():
    """Create a Datasette with a PostgreSQL-backed mutable database."""
    if not POSTGRESQL_TEST_URL:
        pytest.skip("No PostgreSQL configured")
    import psycopg
    from datasette.backends.postgresql import PostgresBackend

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
