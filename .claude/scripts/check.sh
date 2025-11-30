#!/bin/bash
# Run all quality checks

EXIT_CODE=0

echo "=== Type Check ==="
if ! uv run mypy src/; then
    echo "✗ Type check failed"
    EXIT_CODE=1
else
    echo "✓ Type check passed"
fi

echo -e "\n=== Tests ==="
if ! uv run pytest; then
    echo "✗ Tests failed"
    EXIT_CODE=1
else
    echo "✓ Tests passed"
fi

echo -e "\n=== Lint ==="
if ! uv run ruff check .; then
    echo "✗ Lint failed"
    EXIT_CODE=1
else
    echo "✓ Lint passed"
fi

echo -e "\n=== Format ==="
if ! uv run ruff format --check .; then
    echo "✗ Format check failed"
    EXIT_CODE=1
else
    echo "✓ Format check passed"
fi

exit $EXIT_CODE
