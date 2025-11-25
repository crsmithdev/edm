<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Task Execution

- Default to parallelizing work via subagents when tasks are independent
- Spawn subagents for: research, file exploration, testing, implementation of separate components
- Use Task tool for any operation that can run concurrently with other work
- Before starting multi-part work, identify parallelizable branches and spawn subagents for each
- Subagent selection:
  - Research/exploration: Haiku (fast, cheap)
  - Implementation/complex reasoning: Sonnet
  - Only escalate to Opus for architectural decisions or difficult debugging
- Don't wait for one subagent to finish before spawning others if tasks are independent

## Development Context

- Follow instructions in `README.md` to install the package and dependencies
- Music files for testing are available in `~/music`
- Run the CLI through `uv`: `uv run edm ...`
- Always run the CLI with the `--no-color` `--log-level debug`

# Interaction Style

- No preambles, affirmations, or filler ("Great question!", "Sure!", "I'd be happy to")
- No summarizing what you're about to do—just do it
- No sign-offs or offers to help further
- Answer directly, then stop
- When asked to show/generate something, output it immediately without explanation unless asked

# Git Commits

- Subject line only, no body
- 50 characters max
- Lowercase, no period
- Imperative mood: "add feature" not "added feature"

# Code Style

- **Formatter/Linter**: Ruff (`ruff check --fix . && ruff format .`)
- **Line length**: 88 characters
- **Quotes**: Double quotes preferred
- **Imports**: stdlib → third-party → local, sorted alphabetically within groups

# Type Hints

- Always use type hints for function signatures
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- Use `TypeAlias` for complex types to improve readability
- Run `mypy --strict` for type checking

# Project Structure

Use `src/` layout with `pyproject.toml` for packaging. Keep `__init__.py` minimal. Group tests in `tests/` mirroring src structure.

# Dependencies

- **Preferred**: uv (fast) or pip with venv
- Define all deps in `pyproject.toml` under `[project.dependencies]`
- Dev deps go in `[project.optional-dependencies.dev]`

# Testing

- Use pytest with fixtures in `conftest.py`
- Name test files `test_*.py`, test functions `test_*`
- Group related tests in classes prefixed with `Test`
- Use `pytest.raises` for exception testing with `match=` for message validation

# Error Handling

- Create a base `ProjectError` exception, derive specific exceptions from it
- Be specific in except clauses—never bare `except:`
- Use `raise ... from e` to preserve exception chains

# Async

- Prefer `httpx` over `requests` for HTTP (supports async)
- Use `asyncio.TaskGroup` (3.11+) for concurrent tasks
- Avoid mixing sync and async code paths

# Data Validation

- Use Pydantic v2 for data models and validation
- Prefer `model_validator` over `root_validator`
- Use `Field()` for constraints and documentation

# Logging

- Use `logging` stdlib, configure once at entry point
- Use `logger = logging.getLogger(__name__)` per module
- Prefer f-strings in log calls only when level is enabled (or use lazy %)
- Log in natural language (prefer "start bpm analysis" over "start_bpm_analysis")
- Use appropriate and consistent capitalization and punctuation in log messages

# Documentation

- Docstrings: Google style, on public functions/classes
- Keep docstrings to one line when possible
- Type hints replace type info in docstrings

# Commands Reference

```bash
uv sync                                       # Install dependencies
uv run edm analyze <args>                     # Analyze audio files
uv run edm evaluate <args>                    # Evaluate accuracy
uv run ruff check --fix . && ruff format .    # Lint and format
uv run mypy src/                              # Type check
uv run pytest -v                              # Run tests
uv run pytest --cov=src --cov-report=term     # Tests with coverage
```