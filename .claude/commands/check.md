Run all quality checks (format, lint, type check, tests) in sequence.

```bash
set -e
echo "Running format check..."
uv run ruff format .
echo "Running lint..."
uv run ruff check --fix .
echo "Running type check..."
uv run mypy src/
echo "Running tests..."
uv run pytest -v
echo "✓ All quality checks passed!"
```
