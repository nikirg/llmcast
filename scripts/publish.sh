#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running checks..."
uvx ruff check llmcast/ tests/
uvx ruff format --check llmcast/ tests/
uv run ty check llmcast/
uv run pytest

echo "Building..."
rm -rf dist/
uv build

echo "Publishing to PyPI..."
uv publish
