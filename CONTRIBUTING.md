# Contributing

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/edm.git
cd edm

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify installation
uv run edm --version
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates

### Making Changes

1. Create a branch from `main`
2. Make focused, atomic commits
3. Write tests for new functionality
4. Run all checks before committing
5. Open a pull request

### Commit Messages

- Subject line only, no body
- 50 characters max
- Lowercase, no period
- Imperative mood: "add feature" not "added feature"

Examples:
```
add bpm detection for mp4 files
fix structure analysis timeout
refactor audio caching logic
```

## Code Quality

### Before Committing

```bash
# Run all checks
uv run pytest                 # Tests pass
uv run mypy src/              # Types check
uv run ruff check .           # Linting passes
uv run ruff format .          # Code formatted
```

Or use the justfile:
```bash
just check
```

### Style Guide

See [docs/python-style.md](docs/python-style.md) for detailed conventions.

Key points:
- Google-style docstrings
- Type hints on public functions
- Line length: 100 characters
- Use ruff for formatting

### Testing

See [docs/testing.md](docs/testing.md) for test conventions.

- Write tests for new features
- Maintain existing test coverage
- Use parameterized tests for multiple cases
- Put unit tests in `tests/unit/`

## Pull Requests

### Checklist

- [ ] Tests pass locally (`uv run pytest`)
- [ ] Type checks pass (`uv run mypy src/`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Code is formatted (`uv run ruff format .`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

### PR Description

Include:
- Summary of changes
- Related issues (if any)
- Test plan

## Architecture Decisions

For significant changes, create an OpenSpec proposal:

1. Copy template from `openspec/template.md`
2. Create proposal in `openspec/changes/`
3. Prefix with unique code: `[CODE]Proposal Name`
4. Discuss and iterate before implementation

See existing proposals for examples.

## Questions?

- Check [docs/](docs/) for existing documentation
- Review [docs/architecture.md](docs/architecture.md) for system design
- Open an issue for discussion
