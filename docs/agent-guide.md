# Agent Guide

Navigation index for AI agents working in this codebase.

## Quick Reference by Task

### Analyze or Modify BPM Detection

Read together:
- `src/edm/analysis/bpm.py` - Cascading strategy (metadata → spotify → computed)
- `src/edm/analysis/bpm_detector.py` - Computation (beat_this, librosa)
- `tests/test_analysis/test_bpm.py` - Test cases

Key functions:
- `analyze_bpm()` at `src/edm/analysis/bpm.py:40`
- `compute_bpm()` at `src/edm/analysis/bpm_detector.py:224`

### Add or Modify External Service Integration

Read together:
- `src/edm/external/` - Existing services (spotify, beatport, tunebat)
- `src/edm/config.py` - Credential handling
- [architecture.md](architecture.md) - Service integration pattern

### Work on Evaluation Framework

Read together:
- `src/edm/evaluation/common.py` - Shared utilities, metrics
- `src/edm/evaluation/reference.py` - Reference data sources
- `src/edm/evaluation/evaluators/bpm.py` - BPM evaluator
- [cli-reference.md](cli-reference.md) - Evaluation commands

### Modify CLI Commands

Read:
- `src/cli/main.py` - Typer app, command definitions
- [cli-reference.md](cli-reference.md) - Complete CLI documentation

### Work on Logging

Read:
- `src/edm/logging.py` - Structlog configuration
- [development.md](development.md) - Logging patterns

### Run Tests or Add Test Coverage

Read:
- `tests/conftest.py` - Shared fixtures
- [testing.md](testing.md) - Test conventions
- [development.md](development.md) - Test commands

## Documentation Index

| Topic | File | Contents |
|-------|------|----------|
| System design | [architecture.md](architecture.md) | Module organization, data flow, design decisions |
| CLI usage | [cli-reference.md](cli-reference.md) | Commands, options, examples |
| Development | [development.md](development.md) | Setup, testing, logging, code quality |
| Testing | [testing.md](testing.md) | Test framework, conventions |
| Project structure | [project-structure.md](project-structure.md) | Directory layout |
| Python style | [python-style.md](python-style.md) | Code conventions |

## Code Locations by Feature

### Analysis

| Feature | File | Entry Point |
|---------|------|-------------|
| BPM detection | `src/edm/analysis/bpm.py` | `analyze_bpm()` |
| BPM computation | `src/edm/analysis/bpm_detector.py` | `compute_bpm()` |
| Structure detection | `src/edm/analysis/structure.py` | - |

### External Services

| Service | File | Client Class |
|---------|------|--------------|
| Spotify | `src/edm/external/spotify.py` | `SpotifyClient` |
| Beatport | `src/edm/external/beatport.py` | - |
| TuneBat | `src/edm/external/tunebat.py` | - |

### Evaluation

| Component | File | Key Functions |
|-----------|------|---------------|
| Common utilities | `src/edm/evaluation/common.py` | `calculate_mae()`, `calculate_rmse()` |
| Reference sources | `src/edm/evaluation/reference.py` | Reference loading |
| BPM evaluator | `src/edm/evaluation/evaluators/bpm.py` | BPM accuracy evaluation |

### Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| Configuration | `src/edm/config.py` | Pydantic config, env vars |
| Logging | `src/edm/logging.py` | Structlog setup |
| Exceptions | `src/edm/exceptions.py` | Custom exceptions |
| Metadata I/O | `src/edm/io/metadata.py` | Audio file metadata |

## Common Commands

```bash
# Development
uv sync                                      # Install dependencies
uv run edm --help                            # CLI help

# Testing
uv run pytest -v                             # Run tests
uv run pytest --cov=src --cov-report=term    # With coverage

# Code quality
uv run ruff check --fix . && ruff format .   # Lint and format
uv run mypy src/                             # Type check

# Analysis
uv run edm analyze track.mp3 --log-level DEBUG --no-color

# Evaluation
uv run edm evaluate bpm --source ~/music --reference metadata
```

## Task Management

- **OpenSpec**: Features, breaking changes, architecture → `openspec/`
- **TODO.md**: Small tasks and quick fixes (< 1 hour)
- **IDEAS.md**: Improvement ideas from reviews

See `CLAUDE.md` for detailed guidance.
