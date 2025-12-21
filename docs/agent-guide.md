# Agent Guide

Navigation index for AI agents working in this codebase.

## Quick Reference by Task

### Train or Modify ML Models

**Quick start**:
```bash
just train-quick  # 10 epoch test
just train-standard  # 50 epoch production
```

Read together:
- [cheatsheets/training.md](cheatsheets/training.md) - Quick reference for training commands
- [guides/training.md](guides/training.md) - Complete training documentation
- `configs/training_first_run.yaml` - Example configuration
- `packages/edm-lib/src/edm/training/trainer.py` - Training loop implementation
- `packages/edm-lib/src/edm/models/multitask.py` - Model architecture

Key commands:
```bash
# Test training pipeline
just train-quick

# Standard production training
just train-standard

# Monitor training
tensorboard --logdir outputs/training/

# Using config file
uv run edm train --config configs/training_first_run.yaml

# Resume from checkpoint
just train-resume outputs/training/run_xyz/checkpoints/epoch_20.pt
```

Common locations:
- Training configs: `configs/*.yaml`
- Model architectures: `packages/edm-lib/src/edm/models/`
- Training logic: `packages/edm-lib/src/edm/training/`
- Loss functions: `packages/edm-lib/src/edm/training/losses.py`
- Dataset loading: `packages/edm-lib/src/edm/training/dataset.py`

### Analyze or Modify BPM Detection

Read together:
- `packages/edm-lib/src/edm/analysis/bpm.py` - Cascading strategy (metadata → computed)
- `packages/edm-lib/src/edm/analysis/bpm_detector.py` - Computation (beat_this, librosa)
- `packages/edm-lib/tests/unit/test_analysis.py` - BPM analysis tests
- `packages/edm-lib/tests/unit/test_bpm_detector.py` - BPM detector tests

Key functions:
- `analyze_bpm()` at `packages/edm-lib/src/edm/analysis/bpm.py:40`
- `compute_bpm()` at `packages/edm-lib/src/edm/analysis/bpm_detector.py:224`

### Work on Evaluation Framework

Read together:
- `packages/edm-lib/src/edm/evaluation/common.py` - Shared utilities, metrics
- `packages/edm-lib/src/edm/evaluation/reference.py` - Reference data sources
- `packages/edm-lib/src/edm/evaluation/evaluators/bpm.py` - BPM evaluator
- [reference/cli.md](reference/cli.md) - Evaluation commands

### Modify CLI Commands

Read:
- `packages/edm-cli/src/edm_cli/main.py` - Typer app, command definitions
- [reference/cli.md](reference/cli.md) - Complete CLI documentation

### Work on Logging

Read:
- `packages/edm-lib/src/edm/logging.py` - Structlog configuration
- [development.md](development.md) - Logging patterns

### Run Tests or Add Test Coverage

Read:
- `packages/edm-lib/tests/unit/` - Unit tests organized by module
- `tests/fixtures/` - Shared test audio files
- [testing.md](testing.md) - Test conventions
- [development.md](development.md) - Test commands

## Documentation Index

| Topic | File | Contents |
|-------|------|----------|
| System design | [architecture.md](architecture.md) | Module organization, data flow, design decisions |
| CLI usage | [reference/cli.md](reference/cli.md) | Commands, options, examples |
| Development | [development.md](development.md) | Setup, testing, logging, code quality |
| Testing | [testing.md](testing.md) | Test framework, conventions |
| Project structure | [project-structure.md](project-structure.md) | Directory layout |
| Python style | [development/code-style-python.md](development/code-style-python.md) | Code conventions |

## Code Locations by Feature

### Analysis

| Feature | File | Entry Point |
|---------|------|-------------|
| BPM detection | `packages/edm-lib/src/edm/analysis/bpm.py` | `analyze_bpm()` |
| BPM computation | `packages/edm-lib/src/edm/analysis/bpm_detector.py` | `compute_bpm()` |
| Structure detection | `packages/edm-lib/src/edm/analysis/structure.py` | `analyze_structure()` |

### Evaluation

| Component | File | Key Functions |
|-----------|------|---------------|
| Common utilities | `packages/edm-lib/src/edm/evaluation/common.py` | `calculate_mae()`, `calculate_rmse()` |
| Reference sources | `packages/edm-lib/src/edm/evaluation/reference.py` | Reference loading |
| BPM evaluator | `packages/edm-lib/src/edm/evaluation/evaluators/bpm.py` | BPM accuracy evaluation |

### Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| Configuration | `packages/edm-lib/src/edm/config.py` | Pydantic config, env vars |
| Logging | `packages/edm-lib/src/edm/logging.py` | Structlog setup |
| Exceptions | `packages/edm-lib/src/edm/exceptions.py` | Custom exceptions |
| Metadata I/O | `packages/edm-lib/src/edm/io/metadata.py` | Audio file metadata |

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
uv run mypy packages/edm-lib/src/            # Type check

# Analysis
uv run edm analyze track.mp3 --log-level DEBUG --no-color

# Evaluation
uv run edm evaluate bpm --source ~/music --reference metadata
```

## Task Management

- **OpenSpec**: Features, breaking changes, architecture → `openspec/`
- **GitHub Issues**: Bug reports and feature requests

See `CLAUDE.md` for detailed guidance.
