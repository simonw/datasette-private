"""CLI tests for datasette-postgresql plugin.

Requires a PostgreSQL server. Set DATASETTE_TEST_POSTGRESQL=postgresql://...
to enable these tests.
"""

import json
import os

import pytest

from click.testing import CliRunner
from datasette.cli import cli


POSTGRESQL_TEST_URL = os.environ.get("DATASETTE_TEST_POSTGRESQL")

requires_postgresql = pytest.mark.skipif(
    not POSTGRESQL_TEST_URL,
    reason="Set DATASETTE_TEST_POSTGRESQL=postgresql://... to run PostgreSQL tests",
)


@requires_postgresql
def test_cli_connection_string_postgresql():
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
    data = json.loads(result.output)
    # Should have a database with the name from the connection string
    assert len(data.get("databases", [])) > 0


@requires_postgresql
def test_cli_connection_string_postgresql_requires_preview():
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
