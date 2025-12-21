# Development Setup

Setup guide for EDM contributors.

## Prerequisites

- Python 3.12+
- ffmpeg (required for audio loading)
- System packages (Ubuntu/Debian): `python3-dev`, `build-essential`
- Node.js 18+ (for annotator frontend)
- pnpm (Node package manager)

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository

```bash
git clone https://github.com/crsmithdev/edm.git
cd edm
```

### 3. Install Dependencies

```bash
# Install Python dependencies
uv sync

# Install frontend dependencies (if working on annotator)
cd packages/edm-annotator/frontend
pnpm install
cd ../../..
```

### 4. Verify Installation

```bash
# Check CLI works
uv run edm --version

# Check tests pass
uv run pytest

# Check linting
uv run ruff check .
```

## Development Environment

### Python Environment

uv automatically manages Python environments. No need for manual virtual env creation.

```bash
# Run commands in uv environment
uv run python script.py
uv run pytest
uv run edm analyze track.mp3

# Or activate shell (optional)
source .venv/bin/activate
```

### IDE Setup

**VS Code** (recommended):

1. Install Python extension
2. Select interpreter: `.venv/bin/python`
3. Install Ruff extension for linting
4. Install Mypy extension for type checking

**PyCharm**:

1. Set interpreter to `.venv/bin/python`
2. Enable pytest as test runner
3. Configure Ruff as external tool

### Pre-commit Hooks

Install pre-commit hooks for automatic checks:

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Running Tests

```bash
# All tests
uv run pytest

# Verbose output
uv run pytest -v

# Specific test file
uv run pytest tests/unit/test_bpm_detector.py

# Specific test
uv run pytest tests/unit/test_bpm_detector.py::test_compute_bpm_returns_result

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Just commands
just test        # All tests
just test-cov    # With coverage
```

For complete testing guide, see [Testing Guide](testing.md).

## Code Quality

### Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
# Run mypy
uv run mypy src/

# Ignore missing imports (for untyped libraries)
uv run mypy src/ --ignore-missing-imports
```

### Running All Checks

```bash
# Run all quality checks
uv run pytest && uv run mypy src/ && uv run ruff check .

# Or use just command
just check
```

For complete code style guides, see:
- [Python Style Guide](code-style-python.md)
- [JavaScript Style Guide](code-style-javascript.md)

## Development Servers

See [Development Guide](../development.md#development-servers) for running the annotator dev servers.

## Working with Packages

### Package Structure

```
edm/
├── packages/
│   ├── edm-lib/          # Core library
│   ├── edm-cli/          # CLI tool
│   └── edm-annotator/    # Web app
├── pyproject.toml        # Workspace config
└── uv.lock               # Lockfile
```

### Adding Dependencies

```bash
# Add to main project
uv add package-name

# Add dev dependency
uv add --dev package-name

# Add to specific package
cd packages/edm-lib
uv add package-name
```

### Updating Dependencies

```bash
# Update all
uv lock --upgrade
uv sync

# Update specific package
uv add package-name@latest
```

## Git Workflow

### Branch Naming

```bash
# Feature branches
git checkout -b feature/my-feature

# Bug fixes
git checkout -b fix/bug-description

# Documentation
git checkout -b docs/topic
```

### Commit Messages

Follow conventional commits:

```bash
# Good
git commit -m "add bpm detection caching"
git commit -m "fix structure detector boundary alignment"
git commit -m "update training documentation"

# Bad
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "Final version!!!!"
```

Guidelines:
- Lowercase, imperative mood
- 50 characters max
- No period at end
- Describe what the commit does, not what you did

### Pull Requests

1. Create feature branch
2. Make changes with tests
3. Run all checks: `just check`
4. Commit changes
5. Push and create PR
6. Address review feedback
7. Squash and merge when approved

## Debugging

### Verbose Logging

```bash
# CLI with debug output
uv run edm analyze track.mp3 --log-level DEBUG

# Log to file
uv run edm analyze track.mp3 --log-file debug.log --json-logs
```

### Python Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in (Python 3.7+)
breakpoint()
```

### Profiling

```bash
# Profile Python code
uv run python -m cProfile -o profile.stats -m edm_cli.main analyze track.mp3

# View results
uv run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

## Troubleshooting

### `ModuleNotFoundError`

```bash
# Reinstall dependencies
uv sync
```

### Tests Failing

```bash
# Update dependencies
uv sync

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name '*.pyc' -delete

# Run tests again
uv run pytest
```

### IDE Not Finding Modules

```bash
# Verify venv path
ls .venv/bin/python

# Rebuild venv if needed
rm -rf .venv
uv sync
```

For more troubleshooting, see [Troubleshooting Guide](../reference/troubleshooting.md).

## See Also

- **[Testing Guide](testing.md)** - Test framework and patterns
- **[Python Style Guide](code-style-python.md)** - Code conventions
- **[JavaScript Style Guide](code-style-javascript.md)** - Frontend conventions
- **[Development Guide](../development.md)** - Dev servers and workflows
- **[Contributing Guide](../../CONTRIBUTING.md)** - Contribution guidelines
