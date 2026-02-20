#!/bin/bash
# Run tests for the datasette-postgresql plugin.
#
# This script installs datasette from the local repo (not PyPI) so that
# tests always run against the version of datasette in this monorepo.
#
# Usage:
#   DATASETTE_TEST_POSTGRESQL=postgresql://localhost/test ./ci-test.sh
#
# Requirements:
#   - uv must be installed
#   - A PostgreSQL server must be running and DATASETTE_TEST_POSTGRESQL set

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"

echo "Plugin directory: $PLUGIN_DIR"
echo "Repo root: $REPO_ROOT"

# Create a temporary virtual environment
VENV_DIR=$(mktemp -d)
trap "rm -rf $VENV_DIR" EXIT

echo "Creating virtual environment in $VENV_DIR..."
uv venv "$VENV_DIR"

# Install datasette from the repo root (editable so changes are reflected)
echo "Installing datasette from repo..."
uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_ROOT"

# Install the plugin (editable)
echo "Installing datasette-postgresql plugin..."
uv pip install --python "$VENV_DIR/bin/python" -e "$PLUGIN_DIR"

# Install test dependencies
echo "Installing test dependencies..."
uv pip install --python "$VENV_DIR/bin/python" pytest pytest-asyncio

echo "Running tests..."
"$VENV_DIR/bin/python" -m pytest "$PLUGIN_DIR/tests" -v "$@"
