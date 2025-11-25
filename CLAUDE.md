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
  - **Haiku**: Exploration, grep/file searches, running tests, simple refactoring, documentation updates, adding to TODO.md/IDEAS.md
  - **Sonnet**: Feature implementation, bug fixes requiring logic changes, OpenSpec proposals, code reviews, architecture decisions (default choice)
  - **Opus**: Complex architectural changes affecting multiple systems, difficult debugging after Sonnet fails (ONLY when explicitly requested or Sonnet insufficient)
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

# Git Workflow

## Branch Naming
- `feature/<short-name>` - New features
- `fix/<short-name>` - Bug fixes
- `refactor/<short-name>` - Code refactoring
- `docs/<short-name>` - Documentation only

## Commit Granularity
- One commit per logical change
- Tests can be separate commit or included with implementation
- Documentation updates separate from code changes
- Prefer multiple focused commits over single large commits

## When to Push
- After each completed task in tasks.md
- After fixing CI failures
- Before requesting review
- DO NOT push broken tests or failing lints

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

# Quality Checkpoints

## Before Committing
- [ ] All tests pass (`uv run pytest`)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] No TODO comments added without entry in TODO.md
- [ ] Updated relevant docs if architecture/CLI changed

## Before Creating OpenSpec Proposal
- [ ] Read existing specs to check for duplicates (`openspec list --specs`)
- [ ] Read `openspec/project.md` for conventions
- [ ] Run `openspec list` to check for conflicts
- [ ] Validate with `openspec validate --strict` before asking for approval

## Before Marking Task Complete
- [ ] All tasks.md items checked off
- [ ] All code examples in docs still work
- [ ] No commented-out code remains

# Critical Files for Common Tasks

## BPM-related Work
Always read together:
- `src/edm/analysis/bpm.py` (strategy)
- `src/edm/analysis/bpm_detector.py` (implementation)
- `tests/test_analysis/test_bpm.py` (test cases)

## External Services
Always read together:
- `src/edm/external/` (all existing services)
- `src/edm/config.py` (credential handling)
- `docs/architecture.md` (service integration pattern)

## Evaluation Changes
Always read together:
- `src/edm/evaluation/common.py` (shared utilities)
- `src/edm/evaluation/reference.py` (reference sources)
- `docs/cli-reference.md` (evaluation commands)

# Error Recovery Playbooks

## Test Failures
1. Read full test output (don't truncate)
2. Identify root cause (logic error vs test error vs environment)
3. Fix root cause (don't modify tests to pass unless tests are wrong)
4. Re-run affected tests
5. If still failing, re-run full test suite

## Import/Dependency Errors
1. Check if uv.lock is out of sync: `uv sync`
2. Check for circular imports
3. Verify virtual environment is active
4. Check pyproject.toml for missing dependencies

## OpenSpec Validation Failures
1. Run with `--strict` flag for full error details
2. Check scenario format (#### not ### or -)
3. Verify requirement has at least one scenario
4. Use `openspec show <change> --json --deltas-only` to debug

# Agent Workflow Patterns

## Code Review → Implementation
When requesting code review with fixes:
1. Spawn code-reviewer agent for analysis
2. Capture review findings
3. Spawn implementation agent with review context
4. Track both in TodoWrite

## Analysis → Proposal → Implementation
For architecture changes:
1. Spawn explore agent for analysis
2. Return findings and ask for approval
3. Create OpenSpec proposal if approved
4. Implementation only after proposal approval

## Parallel Work
When adding features in separate modules:
- Spawn parallel agents for each module
- Example: "Add Beatport API integration + update evaluation framework"
  - Agent 1: `src/edm/external/beatport.py`
  - Agent 2: `src/edm/evaluation/`

# Task Management

## OpenSpec for Features
Use OpenSpec proposals for:
- New features or capabilities
- Breaking changes
- Architecture changes
- Performance/security work

## TODO.md for Small Tasks
Use TODO.md for:
- Small bugs or quick fixes (< 1 hour)
- Tasks that don't warrant full proposals
- Use `/todo` command for quick capture

## IDEAS.md for Improvements
Use IDEAS.md for:
- Improvement suggestions from code/architecture reviews
- Refactoring opportunities
- Optimizations
- Ideas that might become proposals later
- Use `/idea` command for quick capture

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