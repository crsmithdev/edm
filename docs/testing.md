# Testing Guide

## Framework

- Use pytest with fixtures in `conftest.py`
- Name test files `test_*.py`, test functions `test_*`
- Group related tests in classes prefixed with `Test`
- Use `pytest.raises` for exception testing with `match=` for message validation

## Running Tests

```bash
uv run pytest -v                              # Run tests
uv run pytest --cov=src --cov-report=term     # Tests with coverage
```

## Test Structure

Tests mirror the `src/` directory structure:

```
tests/
├── conftest.py          # Shared fixtures
├── test_analysis/       # Tests for src/edm/analysis/
├── test_evaluation/     # Tests for src/edm/evaluation/
└── ...
```
