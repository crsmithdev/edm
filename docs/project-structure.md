# Project Structure

## Layout

Use `src/` layout with `pyproject.toml` for packaging. Keep `__init__.py` minimal. Group tests in `tests/` mirroring src structure.

```
edm/
├── src/
│   ├── cli/             # CLI entry point and commands
│   └── edm/             # Core library
│       ├── analysis/    # Audio analysis modules
│       ├── evaluation/  # Accuracy evaluation
│       ├── external/    # External service integrations
│       ├── features/    # Feature extraction
│       ├── io/          # File I/O operations
│       └── models/      # Data models
├── tests/               # Test suite
├── docs/                # Documentation
└── openspec/            # Change proposals
```

## Dependencies

- **Preferred**: uv (fast) or pip with venv
- Define all deps in `pyproject.toml` under `[project.dependencies]`
- Dev deps go in `[project.optional-dependencies.dev]`
