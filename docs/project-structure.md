# Project Structure

## Layout

```
edm/
├── src/
│   ├── cli/                 # CLI entry point and commands
│   │   ├── main.py         # Typer app with subcommands
│   │   └── commands/       # Command implementations
│   │       ├── analyze.py  # analyze command
│   │       └── evaluate.py # evaluate command
│   └── edm/                 # Core library
│       ├── analysis/        # Audio analysis modules
│       │   ├── bpm.py      # BPM public API
│       │   ├── bpm_detector.py # BPM detectors
│       │   ├── bars.py     # Bar calculations
│       │   ├── beat_detector.py # Beat tracking
│       │   ├── beat_grid.py # Beat grid alignment
│       │   ├── structure.py # Structure public API
│       │   └── structure_detector.py # Structure detectors
│       ├── data/            # Data management
│       │   ├── converters.py # Format converters
│       │   ├── export.py   # Export utilities
│       │   ├── metadata.py # Metadata models
│       │   ├── rekordbox.py # Rekordbox integration
│       │   ├── schema.py   # Data schemas
│       │   └── validation.py # Validation utilities
│       ├── evaluation/      # Accuracy evaluation
│       │   ├── common.py   # Shared utilities
│       │   ├── reference.py # Reference data sources
│       │   └── evaluators/ # Evaluation implementations
│       ├── io/              # File I/O operations
│       │   ├── audio.py    # Audio loading/caching
│       │   ├── files.py    # File utilities
│       │   └── metadata.py # Metadata I/O
│       ├── models/          # ML models
│       │   ├── backbone.py # Model backbones
│       │   ├── heads.py    # Task heads
│       │   └── multitask.py # Multitask model
│       ├── processing/      # Processing utilities
│       │   └── parallel.py # Parallel processing
│       ├── config.py        # Configuration
│       ├── exceptions.py    # Custom exceptions
│       └── logging.py       # Structlog setup
├── tests/                   # Test suite
│   ├── fixtures/           # Test audio files
│   ├── estimations/        # Reference annotations
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── performance/        # Performance benchmarks
├── docs/                    # Documentation
├── data/                    # Data files (gitignored)
│   ├── annotations/        # Track annotations
│   └── accuracy/           # Evaluation results
└── openspec/                # Change proposals
    ├── changes/            # Active proposals
    └── archive/            # Completed proposals
```

## Key Directories

### `src/edm/analysis/`

Core audio analysis with two-tier architecture:
- **Public APIs** (`bpm.py`, `structure.py`): Stable interfaces
- **Detectors** (`bpm_detector.py`, `structure_detector.py`): Algorithm implementations

### `src/edm/data/`

Data management layer:
- Schema definitions and validation
- Format converters (JAMS, Rekordbox XML)
- Export utilities

### `src/edm/models/`

Machine learning models:
- Backbone architectures (CNN, etc.)
- Task-specific heads
- Multitask training infrastructure

### `src/edm/evaluation/`

Accuracy evaluation framework:
- Reference data loading
- Metric calculations (MAE, RMSE, precision/recall)
- Per-evaluator implementations

### `tests/`

Organized by test type:
- `unit/`: Fast, isolated tests
- `integration/`: Component interaction tests
- `performance/`: Benchmarks and speedup tests

## Dependencies

- **Package manager**: uv (recommended) or pip
- All dependencies in `pyproject.toml` under `[project.dependencies]`
- Dev dependencies in `[project.optional-dependencies.dev]`

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, dependencies, tools |
| `.python-version` | Python version (3.12) |
| `justfile` | Task runner commands |
| `.pre-commit-config.yaml` | Pre-commit hooks |
