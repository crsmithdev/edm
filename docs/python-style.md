# Python Style Guide

## Code Style

- **Formatter/Linter**: Ruff (`ruff check --fix . && ruff format .`)
- **Line length**: 88 characters
- **Quotes**: Double quotes preferred
- **Imports**: stdlib → third-party → local, sorted alphabetically within groups

## Type Hints

- Always use type hints for function signatures
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- Use `TypeAlias` for complex types to improve readability
- Run `mypy --strict` for type checking

## Error Handling

- Create a base `ProjectError` exception, derive specific exceptions from it
- Be specific in except clauses—never bare `except:`
- Use `raise ... from e` to preserve exception chains

## Async

- Prefer `httpx` over `requests` for HTTP (supports async)
- Use `asyncio.TaskGroup` (3.11+) for concurrent tasks
- Avoid mixing sync and async code paths

## Data Validation

- Use Pydantic v2 for data models and validation
- Prefer `model_validator` over `root_validator`
- Use `Field()` for constraints and documentation

## Logging

- Use `logging` stdlib, configure once at entry point
- Use `logger = logging.getLogger(__name__)` per module
- Prefer f-strings in log calls only when level is enabled (or use lazy %)
- Log in natural language (prefer "start bpm analysis" over "start_bpm_analysis")
- Use appropriate and consistent capitalization and punctuation in log messages

## Documentation

- Docstrings: Google style, on public functions/classes
- Keep docstrings to one line when possible
- Type hints replace type info in docstrings
