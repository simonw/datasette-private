from datasette import hookimpl

from .backend import PostgresBackend


@hookimpl
def register_database_backends(datasette):
    return [PostgresBackend]
