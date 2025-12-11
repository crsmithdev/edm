# Project Structure

## Monorepo Layout

This project uses a 3-package monorepo managed with uv workspaces:

```
edm/
├── packages/
│   ├── edm-lib/           Core library (analysis, training, models)
│   ├── edm-cli/           Command-line interface
│   └── edm-annotator/     Web-based annotation tool
│
├── docs/                  Documentation
├── data/                  Data files (gitignored)
│   ├── annotations/      Track annotations (YAML)
│   └── accuracy/         Evaluation results
│
├── tests/                 Shared test fixtures
│   ├── fixtures/         Test audio files
│   └── estimations/      Reference annotations
│
├── configs/              Training configuration files
├── scripts/              Utility scripts
├── openspec/             Change proposals
│   ├── changes/         Active proposals
│   └── archive/         Completed proposals
│
├── pyproject.toml        Workspace configuration
├── uv.lock              Single lockfile for all packages
└── justfile             Task automation
```

## Package: `edm-lib`

Core library for audio analysis, ML models, and training.

```
packages/edm-lib/
├── src/edm/                    Library code
│   ├── analysis/              Audio analysis modules
│   │   ├── bpm.py            BPM public API
│   │   ├── bpm_detector.py   BPM detector implementations
│   │   ├── structure.py      Structure public API
│   │   ├── structure_detector.py   Structure detectors
│   │   ├── ml_detector.py    ML-based structure detection
│   │   ├── beat_detector.py  Beat tracking
│   │   ├── beat_grid.py      Beat grid alignment
│   │   ├── bars.py           Bar calculations
│   │   ├── orchestrator.py   Analysis coordination
│   │   └── validation/       Result validation
│   │       ├── base.py
│   │       ├── beat_structure.py
│   │       ├── downbeat_structure.py
│   │       ├── results.py
│   │       └── orchestrator.py
│   │
│   ├── training/              ML model training
│   │   ├── trainer.py        Training loop
│   │   ├── dataset.py        Data loading
│   │   ├── losses.py         Loss functions
│   │   └── cleanlab_utils.py Data quality utilities
│   │
│   ├── models/                Neural network architectures
│   │   ├── backbone.py       Feature extractors (MERT, CNN)
│   │   ├── heads.py          Task-specific heads
│   │   └── multitask.py      Multi-task model
│   │
│   ├── data/                  Data management
│   │   ├── schema.py         Pydantic data models
│   │   ├── validation.py     Schema validation
│   │   ├── converters.py     Format conversion
│   │   ├── jams_io.py        JAMS format support
│   │   ├── rekordbox.py      Rekordbox XML import
│   │   ├── metadata.py       Track metadata models
│   │   └── export.py         Export utilities
│   │
│   ├── evaluation/            Accuracy evaluation
│   │   ├── common.py         Metrics (MAE, RMSE, F1)
│   │   ├── reference.py      Reference data sources
│   │   ├── metrics.py        Evaluation metrics
│   │   └── evaluators/       Per-feature evaluators
│   │       ├── bpm.py
│   │       └── structure.py
│   │
│   ├── io/                    File I/O operations
│   │   ├── audio.py          Audio loading with LRU cache
│   │   ├── files.py          File utilities
│   │   └── metadata.py       Metadata extraction
│   │
│   ├── processing/            Processing utilities
│   │   └── parallel.py       Parallel execution
│   │
│   ├── registry/              Model registry
│   │   └── mlflow_registry.py   MLflow integration
│   │
│   ├── config.py              Configuration (Pydantic)
│   ├── exceptions.py          Custom exception classes
│   └── logging.py             Structured logging (structlog)
│
├── tests/                     Package-specific tests
│   ├── unit/                 Unit tests by module
│   └── integration/          Integration tests
│
└── pyproject.toml            Package metadata + dependencies
```

**Import as**: `from edm.analysis import analyze_bpm`

## Package: `edm-cli`

Command-line interface built with Typer.

```
packages/edm-cli/
├── src/edm_cli/              CLI code
│   ├── main.py              Typer app entry point
│   └── commands/            Command implementations
│       ├── analyze.py       analyze command - parallel audio analysis
│       ├── train.py         train command - model training
│       ├── evaluate.py      evaluate command - accuracy evaluation
│       ├── data.py          data subcommands - format conversion
│       └── models.py        models subcommands - model management
│
└── pyproject.toml           Package metadata
    [project.scripts]
    edm = "edm_cli.main:app"   # Defines 'edm' command
```

**Commands**:
- `edm analyze` - Analyze audio files
- `edm train` - Train ML models
- `edm evaluate` - Evaluate accuracy
- `edm data` - Data management
- `edm models` - Model management

## Package: `edm-annotator`

Web-based annotation tool with Flask backend + React frontend.

```
packages/edm-annotator/
├── backend/                  Flask API
│   ├── src/edm_annotator/
│   │   ├── app.py           Application factory
│   │   ├── config.py        Environment configs (dev/prod/test)
│   │   ├── api/             Route blueprints
│   │   │   ├── tracks.py
│   │   │   ├── audio.py
│   │   │   ├── waveforms.py
│   │   │   └── annotations.py
│   │   └── services/        Business logic layer
│   │       ├── audio_service.py
│   │       ├── waveform_service.py
│   │       └── annotation_service.py
│   │
│   └── tests/
│       ├── unit/            Service/component tests
│       └── integration/     API endpoint tests
│
├── frontend/                 React + TypeScript UI
│   ├── src/
│   │   ├── stores/          Zustand state management
│   │   │   ├── audioStore.ts
│   │   │   ├── trackStore.ts
│   │   │   ├── waveformStore.ts
│   │   │   ├── structureStore.ts
│   │   │   ├── tempoStore.ts
│   │   │   └── uiStore.ts
│   │   ├── services/        API client
│   │   │   └── api.ts
│   │   ├── utils/           Helper functions
│   │   └── components/      React components
│   │
│   ├── tests/
│   │   ├── unit/           Component tests (vitest)
│   │   └── e2e/            End-to-end tests
│   │
│   ├── package.json        Frontend dependencies (pnpm)
│   ├── tsconfig.json       TypeScript config
│   └── vite.config.ts      Vite build config
│
├── run-dev.sh              Development server launcher
└── pyproject.toml          Backend dependencies
```

**Run**: `just annotator` (starts backend + frontend)
- Backend: http://localhost:5000
- Frontend: http://localhost:5173

## Key Directories

### `docs/`

Documentation files:
- `architecture.md` - System design and module organization
- `cli-reference.md` - Complete command documentation
- `development.md` - Setup, testing, code quality
- `testing.md` - Test framework, conventions
- `training.md` - Model training guide
- `agent-guide.md` - Navigation for AI agents
- `python-style.md` - Python code conventions
- `javascript-style.md` - TypeScript/React conventions
- `edm-terminology.md` - EDM domain knowledge

### `data/`

Data files (gitignored):
- `data/annotations/*.yaml` - Manual track annotations
- `data/accuracy/` - Evaluation results

Audio files stored separately in `~/music` by default.

### `tests/`

Root-level shared test fixtures:
- `tests/fixtures/` - Test audio files
- `tests/estimations/` - Reference annotations for validation

Each package also has its own `tests/` directory for package-specific tests.

### `configs/`

Training configuration files:
- `configs/training_first_run.yaml` - Quick start config
- `configs/*.yaml` - Other training presets

### `scripts/`

Standalone utility scripts:
- `scripts/evaluate_model.py` - Model evaluation
- `scripts/evaluate_model_detailed.py` - Detailed accuracy
- `scripts/benchmark_detectors.py` - Performance benchmarking
- `scripts/detect_label_errors.py` - Data quality analysis
- `scripts/ablation_study.py` - Ablation experiments

### `openspec/`

OpenSpec change management:
- `openspec/changes/` - Active proposals
- `openspec/archive/` - Completed changes
- `openspec/specs/` - System specifications

## Monorepo Commands

```bash
# Install all packages
uv sync

# Run CLI (uv resolves which package provides 'edm')
uv run edm analyze track.mp3

# Run tests for specific package
uv run pytest packages/edm-lib/tests/

# Run tests for all packages
uv run pytest

# Type check specific package
uv run mypy packages/edm-lib/src/

# Lint all packages
uv run ruff check .

# Format all code
uv run ruff format .
```

## Workspace Configuration

**Root `pyproject.toml`**:
```toml
[tool.uv.workspace]
members = ["packages/*"]
```

**Package imports work seamlessly**:
```python
# In edm-cli/src/edm_cli/main.py
from edm.analysis import analyze_bpm  # Imports from edm-lib
```

uv handles cross-package dependencies via the workspace configuration.

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Workspace config, shared tool settings |
| `uv.lock` | Single lockfile for all packages |
| `justfile` | Task automation (check, test, clean, etc.) |
| `.python-version` | Python version (3.12) |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `dvc.yaml` | DVC pipeline for training |

## Tool Configuration

Shared tool configs in root `pyproject.toml`:

- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing framework
- **coverage**: Code coverage

Each package can override settings in its own `pyproject.toml`.
