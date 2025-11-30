#!/bin/bash
# Wipe and reinstall all dependencies from scratch

set -e

echo "=== Cleaning build artifacts ==="
.claude/scripts/clean.sh

echo -e "\n=== Removing virtual environment ==="
rm -rf .venv

echo "=== Removing lock file ==="
rm -f uv.lock

echo "=== Reinstalling dependencies ==="
uv sync

echo -e "\n✓ Dependencies reinstalled"
