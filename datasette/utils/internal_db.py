import textwrap


async def init_internal_db(db):
    create_tables_sql = textwrap.dedent("""
    CREATE TABLE IF NOT EXISTS catalog_databases (
        database_name TEXT PRIMARY KEY,
        path TEXT,
        is_memory INTEGER,
        schema_version INTEGER
    );
    CREATE TABLE IF NOT EXISTS catalog_tables (
        database_name TEXT,
        table_name TEXT,
        rootpage INTEGER,
        sql TEXT,
        PRIMARY KEY (database_name, table_name),
        FOREIGN KEY (database_name) REFERENCES catalog_databases(database_name)
    );
    CREATE TABLE IF NOT EXISTS catalog_views (
        database_name TEXT,
        view_name TEXT,
        rootpage INTEGER,
        sql TEXT,
        PRIMARY KEY (database_name, view_name),
        FOREIGN KEY (database_name) REFERENCES catalog_databases(database_name)
    );
    CREATE TABLE IF NOT EXISTS catalog_columns (
        database_name TEXT,
        table_name TEXT,
        cid INTEGER,
        name TEXT,
        type TEXT,
        "notnull" INTEGER,
        default_value TEXT, -- renamed from dflt_value
        is_pk INTEGER, -- renamed from pk
        hidden INTEGER,
        PRIMARY KEY (database_name, table_name, name),
        FOREIGN KEY (database_name) REFERENCES catalog_databases(database_name),
        FOREIGN KEY (database_name, table_name) REFERENCES catalog_tables(database_name, table_name)
    );
    CREATE TABLE IF NOT EXISTS catalog_indexes (
        database_name TEXT,
        table_name TEXT,
        seq INTEGER,
        name TEXT,
        "unique" INTEGER,
        origin TEXT,
        partial INTEGER,
        PRIMARY KEY (database_name, table_name, name),
        FOREIGN KEY (database_name) REFERENCES catalog_databases(database_name),
        FOREIGN KEY (database_name, table_name) REFERENCES catalog_tables(database_name, table_name)
    );
    CREATE TABLE IF NOT EXISTS catalog_foreign_keys (
        database_name TEXT,
        table_name TEXT,
        id INTEGER,
        seq INTEGER,
        "table" TEXT,
        "from" TEXT,
        "to" TEXT,
        on_update TEXT,
        on_delete TEXT,
        match TEXT,
        PRIMARY KEY (database_name, table_name, id, seq),
        FOREIGN KEY (database_name) REFERENCES catalog_databases(database_name),
        FOREIGN KEY (database_name, table_name) REFERENCES catalog_tables(database_name, table_name)
    );
    """).strip()
    await db.execute_write_script(create_tables_sql)
    await initialize_metadata_tables(db)


async def initialize_metadata_tables(db):
    await db.execute_write_script(textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS metadata_instance (
            key text,
            value text,
            unique(key)
        );

        CREATE TABLE IF NOT EXISTS metadata_databases (
            database_name text,
            key text,
            value text,
            unique(database_name, key)
        );

        CREATE TABLE IF NOT EXISTS metadata_resources (
            database_name text,
            resource_name text,
            key text,
            value text,
            unique(database_name, resource_name, key)
        );

        CREATE TABLE IF NOT EXISTS metadata_columns (
            database_name text,
            resource_name text,
            column_name text,
            key text,
            value text,
            unique(database_name, resource_name, column_name, key)
        );
            """))


async def populate_schema_tables(internal_db, db):
    database_name = db.name
    backend = db.backend

    def delete_everything(conn):
        conn.execute(
            "DELETE FROM catalog_tables WHERE database_name = ?", [database_name]
        )
        conn.execute(
            "DELETE FROM catalog_views WHERE database_name = ?", [database_name]
        )
        conn.execute(
            "DELETE FROM catalog_columns WHERE database_name = ?", [database_name]
        )
        conn.execute(
            "DELETE FROM catalog_foreign_keys WHERE database_name = ?",
            [database_name],
        )
        conn.execute(
            "DELETE FROM catalog_indexes WHERE database_name = ?", [database_name]
        )

    await internal_db.execute_write_fn(delete_everything)

    def collect_info(conn):
        tables_to_insert = []
        views_to_insert = []
        columns_to_insert = []
        foreign_keys_to_insert = []
        indexes_to_insert = []

        # Use backend methods for schema introspection
        table_names = backend.table_names(conn)
        view_names = backend.view_names(conn)

        for view_name in view_names:
            view_def = backend.get_view_definition(conn, view_name)
            views_to_insert.append(
                (database_name, view_name, 0, view_def)
            )

        for table_name in table_names:
            table_def = backend.get_table_definition(conn, table_name)
            tables_to_insert.append(
                (database_name, table_name, 0, table_def)
            )
            columns = backend.table_column_details(conn, table_name)
            columns_to_insert.extend(
                {
                    **{"database_name": database_name, "table_name": table_name},
                    **column._asdict(),
                }
                for column in columns
            )
            fks = backend.foreign_keys_for_table(conn, table_name)
            for i, fk in enumerate(fks):
                foreign_keys_to_insert.append(
                    {
                        "database_name": database_name,
                        "table_name": table_name,
                        "id": i,
                        "seq": 0,
                        "table": fk["other_table"],
                        "from": fk["column"],
                        "to": fk["other_column"],
                        "on_update": "NO ACTION",
                        "on_delete": "NO ACTION",
                        "match": "NONE",
                    }
                )
            indexes = backend.indexes_for_table(conn, table_name)
            for i, index in enumerate(indexes):
                indexes_to_insert.append(
                    {
                        "database_name": database_name,
                        "table_name": table_name,
                        "seq": index.get("seq", i),
                        "name": index.get("name", ""),
                        "unique": index.get("unique", 0),
                        "origin": index.get("origin", ""),
                        "partial": index.get("partial", 0),
                    }
                )
        return (
            tables_to_insert,
            views_to_insert,
            columns_to_insert,
            foreign_keys_to_insert,
            indexes_to_insert,
        )

    (
        tables_to_insert,
        views_to_insert,
        columns_to_insert,
        foreign_keys_to_insert,
        indexes_to_insert,
    ) = await db.execute_fn(collect_info)

    await internal_db.execute_write_many(
        """
        INSERT INTO catalog_tables (database_name, table_name, rootpage, sql)
        values (?, ?, ?, ?)
    """,
        tables_to_insert,
    )
    await internal_db.execute_write_many(
        """
        INSERT INTO catalog_views (database_name, view_name, rootpage, sql)
        values (?, ?, ?, ?)
    """,
        views_to_insert,
    )
    await internal_db.execute_write_many(
        """
        INSERT INTO catalog_columns (
            database_name, table_name, cid, name, type, "notnull", default_value, is_pk, hidden
        ) VALUES (
            :database_name, :table_name, :cid, :name, :type, :notnull, :default_value, :is_pk, :hidden
        )
    """,
        columns_to_insert,
    )
    await internal_db.execute_write_many(
        """
        INSERT INTO catalog_foreign_keys (
            database_name, table_name, "id", seq, "table", "from", "to", on_update, on_delete, match
        ) VALUES (
            :database_name, :table_name, :id, :seq, :table, :from, :to, :on_update, :on_delete, :match
        )
    """,
        foreign_keys_to_insert,
    )
    await internal_db.execute_write_many(
        """
        INSERT INTO catalog_indexes (
            database_name, table_name, seq, name, "unique", origin, partial
        ) VALUES (
            :database_name, :table_name, :seq, :name, :unique, :origin, :partial
        )
    """,
        indexes_to_insert,
    )
