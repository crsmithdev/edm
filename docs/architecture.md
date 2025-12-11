# Architecture

## System Overview

EDM is a monorepo containing three packages that work together to analyze, annotate, and train models for Electronic Dance Music track structure analysis:

```
packages/
├── edm-lib/        Core analysis library and ML training
├── edm-cli/        Command-line interface
└── edm-annotator/  Web-based annotation tool
```

All packages are managed using uv workspaces with a single lockfile (`uv.lock`) for consistent dependencies.

## Design Philosophy

1. **Offline-first**: No external API dependencies; all processing happens locally
2. **Two-tier abstraction**: Public APIs (`bpm.py`, `structure.py`) separate from implementations (`*_detector.py`)
3. **Cascading strategies**: Multiple detection methods with fallback (metadata → neural → traditional)
4. **Graceful degradation**: Structure analysis works even without BPM
5. **Modular architecture**: Clear separation between CLI, library, training, and annotation

## Package Architecture

### `packages/edm-lib/` - Core Library

The analysis and training engine. All core functionality lives here.

```
src/edm/
├── analysis/           Audio analysis modules
│   ├── bpm.py         Public BPM API
│   ├── bpm_detector.py   BPM detection implementations
│   ├── structure.py   Public structure API
│   ├── structure_detector.py   Structure detection implementations
│   ├── ml_detector.py    ML-based structure detection
│   ├── beat_detector.py  Beat tracking
│   ├── beat_grid.py   Beat grid alignment
│   ├── bars.py        Time ↔ bar calculations
│   ├── orchestrator.py   Analysis coordination
│   └── validation/    Result validation
│
├── training/          ML model training
│   ├── trainer.py     Training loop
│   ├── dataset.py     Data loading
│   ├── losses.py      Loss functions
│   └── cleanlab_utils.py   Data quality
│
├── models/            Neural network architectures
│   ├── backbone.py    Feature extractors (MERT, CNN)
│   ├── heads.py       Task-specific heads
│   └── multitask.py   Multi-task model
│
├── data/              Data management
│   ├── schema.py      Pydantic data models
│   ├── validation.py  Schema validation
│   ├── converters.py  Format conversion
│   ├── jams_io.py     JAMS format support
│   ├── rekordbox.py   Rekordbox XML import
│   ├── metadata.py    Track metadata models
│   └── export.py      Export utilities
│
├── evaluation/        Accuracy evaluation
│   ├── common.py      Metrics (MAE, RMSE, F1)
│   ├── reference.py   Reference data sources
│   ├── metrics.py     Evaluation metrics
│   └── evaluators/    Per-feature evaluators
│       ├── bpm.py
│       └── structure.py
│
├── io/                File I/O operations
│   ├── audio.py       Audio loading with LRU cache
│   ├── files.py       File utilities
│   └── metadata.py    Metadata extraction
│
├── processing/        Processing utilities
│   └── parallel.py    Parallel execution
│
├── registry/          Model registry
│   └── mlflow_registry.py   MLflow integration
│
├── config.py          Configuration (Pydantic)
├── exceptions.py      Custom exception classes
└── logging.py         Structured logging (structlog)
```

### `packages/edm-cli/` - CLI Interface

Typer-based command-line interface built on top of `edm-lib`.

```
src/edm_cli/
├── main.py            Typer app entry point
└── commands/          Command implementations
    ├── analyze.py     analyze command - parallel audio analysis
    ├── train.py       train command - model training
    ├── evaluate.py    evaluate command - accuracy evaluation
    ├── data.py        data subcommands - format conversion
    └── models.py      models subcommands - model management
```

**Entry point**: `edm` command (defined in `pyproject.toml`)

### `packages/edm-annotator/` - Annotation Tool

Web-based UI for manual annotation of EDM track structure.

```
edm-annotator/
├── backend/           Flask API
│   └── src/edm_annotator/
│       ├── app.py     Application factory
│       ├── config.py  Environment configs
│       ├── api/       Route blueprints
│       │   ├── tracks.py
│       │   ├── audio.py
│       │   ├── waveforms.py
│       │   └── annotations.py
│       └── services/  Business logic
│           ├── audio_service.py
│           ├── waveform_service.py
│           └── annotation_service.py
│
├── frontend/          React + TypeScript UI
│   └── src/
│       ├── stores/    Zustand state management
│       ├── services/  API client
│       ├── utils/     Helper functions
│       └── components/   React components
│
└── run-dev.sh         Development server launcher
```

**Quick start**: `just annotator` from repo root

## Data Flow

### Analysis Pipeline

```
Audio file (MP3/FLAC/WAV)
    ↓
[Audio Loader] → LRU cache (10 tracks)
    ↓
[Analysis Orchestrator]
    ├→ [BPM Detection] → metadata → beat_this → librosa
    ├→ [Structure Detection] → MSAF → energy-based fallback
    ├→ [Beat Detection] → beat tracking + grid alignment
    └→ [Bar Calculations] → time ↔ bar conversion
    ↓
[Validation] → beat/downbeat validators
    ↓
Results (JSON/YAML/Table)
```

### Training Pipeline

```
Annotations (YAML) + Audio files (~/music)
    ↓
[Dataset] → audio loading + feature extraction
    ↓
[Trainer]
    ├→ [Backbone] → MERT-95M/330M or CNN
    ├→ [Task Heads] → boundary/beat/energy/label
    ├→ [Losses] → BCE + focal loss
    └→ [MLflow] → experiment tracking
    ↓
Checkpoints → outputs/training/{run_id}/
    ↓
[Model Registry] → MLflow model versioning
```

### Annotation Workflow

```
User selects track → Annotator UI
    ↓
[Frontend] → track list, waveform display
    ↓
[Flask API] → /api/tracks, /api/waveforms, /api/annotations
    ↓
[Backend Services]
    ├→ Audio Service → load audio, extract metadata
    ├→ Waveform Service → generate waveform data
    └→ Annotation Service → save/load annotations
    ↓
Annotations saved → data/annotations/{track_id}.yaml
    ↓
Used for training models
```

## Module Organization

### Two-Tier Pattern

Analysis modules use a two-tier architecture for stability:

**Tier 1: Public APIs** (`bpm.py`, `structure.py`)
- Stable interfaces that never change
- Coordinate multiple detection strategies
- Handle caching, validation, error handling
- Return standardized data models

**Tier 2: Detectors** (`bpm_detector.py`, `structure_detector.py`)
- Algorithm implementations
- Can change frequently
- Called by public APIs
- Focus on accuracy and performance

Example:
```python
# Tier 1: Public API
from edm.analysis.bpm import analyze_bpm

result = analyze_bpm(audio_path)  # Stable interface

# Tier 2: Detector (internal use only)
from edm.analysis.bpm_detector import compute_bpm

bpm = compute_bpm(audio_data)  # Implementation detail
```

### Cascading Strategies

BPM detection uses cascading strategies for speed vs accuracy tradeoff:

1. **Metadata** (fastest, ~1ms)
   - Read from ID3/FLAC/MP4 tags
   - Skip computation if confident

2. **beat_this** (accurate, ~10-15s)
   - Neural network beat tracker
   - Handles tempo multiplicity
   - Best for EDM genre

3. **librosa** (fallback, ~5-8s)
   - Traditional spectral flux + autocorrelation
   - Reliable baseline

Structure detection similarly cascades:
1. **MSAF** - Spectral flux boundaries + energy labeling
2. **Energy-based** - RMS + spectral contrast fallback

## Design Decisions

### Why Monorepo?

- **Single lockfile**: All packages use same dependency versions
- **Cross-package imports**: `edm-cli` imports from `edm-lib` seamlessly
- **Unified testing**: Run all tests with single command
- **Atomic changes**: Update library + CLI together in one commit

### Why Two-Tier Abstraction?

- **API stability**: Public APIs don't break when algorithms change
- **Flexibility**: Easy to swap detector implementations
- **Testing**: Mock detectors without touching public API
- **Backwards compatibility**: Old code continues working

### Why Cascading Detection?

- **Performance**: Metadata BPM is instant; skip expensive computation
- **Accuracy**: Neural networks for hard cases
- **Reliability**: Traditional methods as fallback
- **User control**: `--no-metadata` flag to force computation

### Why LRU Caching?

- **Batch processing**: Analyzing 50+ files reuses loaded audio
- **Memory bounded**: 10 track limit (~2GB)
- **Thread-safe**: Works with parallel processing
- **Automatic**: No manual cache management

### Why Pydantic for Schemas?

- **Type safety**: Runtime validation + static typing
- **IDE support**: Autocomplete for all fields
- **Serialization**: JSON/YAML conversion built-in
- **Documentation**: Schema self-documenting

### Why structlog for Logging?

- **Structured output**: JSON logs for production
- **Context binding**: Track request IDs through call chain
- **Performance**: Lazy evaluation, minimal overhead
- **Flexibility**: Console (human) + JSON (machine) outputs

## Performance Characteristics

### CPU-Bound Processing

BPM detection is CPU-bound (neural network inference). Parallelism scales linearly:

| Workers | Files | Time | Speedup |
|---------|-------|------|---------|
| 1 | 50 | ~20 min | 1x |
| 4 | 50 | ~5 min | 4x |
| 8 | 50 | ~3 min | 6-7x |

Default: `CPU_count - 1` workers (leaves 1 core for OS)

### Memory Requirements

- **Per worker**: ~200MB (audio file in memory)
- **Audio cache**: ~2GB (10 tracks × 200MB)
- **Total (8 workers)**: ~3-4GB typical usage

Adjust `--workers` flag if memory constrained.

### Bottlenecks

1. **BPM computation** (10-15s/track) - neural network inference
2. **Audio loading** (2-3s/track) - I/O + decoding
3. **Structure detection** (3-5s/track) - MSAF segmentation

Cache mitigates (2) for batch processing.

## Extension Points

### Adding a New Detector

1. Implement detector in `edm-lib/src/edm/analysis/*_detector.py`
2. Add to cascading strategy in public API
3. Add CLI flag if user-selectable
4. Add tests in `tests/unit/`

### Adding a New Analysis Type

1. Create `src/edm/analysis/{feature}.py` (public API)
2. Create `src/edm/analysis/{feature}_detector.py` (implementation)
3. Add to `AnalysisOrchestrator`
4. Add CLI command in `edm-cli/src/edm_cli/commands/analyze.py`

### Adding a New Model Architecture

1. Add backbone to `src/edm/models/backbone.py`
2. Add task head to `src/edm/models/heads.py`
3. Update config schema in training docs
4. Add to `configs/` directory

### Adding a New Output Format

1. Implement exporter in `src/edm/data/export.py`
2. Add format option to CLI
3. Add tests for serialization/deserialization

## Configuration

### Environment Variables

- `AUDIO_DIR` - Default audio directory (default: `~/music`)
- `ANNOTATION_DIR` - Annotation storage (default: `data/annotations`)
- `LOG_LEVEL` - Logging verbosity (default: `INFO`)

### Config Files

- `pyproject.toml` - Package metadata + dependencies
- `uv.lock` - Locked dependency versions
- `configs/*.yaml` - Training configurations
- `.claude/` - AI assistant contexts and commands

## Testing Strategy

- **Unit tests** (`tests/unit/`) - Fast, isolated module tests
- **Integration tests** (`tests/integration/`) - Component interaction
- **Performance tests** (`tests/performance/`) - Benchmarks
- **Fixtures** (`tests/fixtures/`) - Test audio files

Run all: `uv run pytest -v`

## Dependencies

### Core Analysis
- `librosa >= 0.10.0` - Audio processing
- `beat_this` - Neural beat tracking
- `msaf >= 0.1.80` - Music segmentation

### Deep Learning
- `torch >= 2.0.0` - Neural networks
- `transformers >= 4.30.0` - MERT models

### CLI & UI
- `typer >= 0.9.0` - CLI framework
- `rich >= 13.0.0` - Terminal formatting

### Data & Validation
- `pydantic >= 2.0.0` - Schema validation
- `pyyaml >= 6.0.0` - YAML parsing

### Experiment Tracking
- `mlflow >= 2.10.0` - Experiment tracking
- `tensorboard >= 2.0.0` - Visualization
- `dvc >= 3.0.0` - Data versioning

See `packages/*/pyproject.toml` for complete dependency lists.

## Future Enhancements

### Planned Features
- Cross-validation orchestrator (currently placeholder)
- Real-time analysis mode
- GPU acceleration for training
- Web API server mode

### Known Limitations
- Config file loading recognized but not implemented
- Evaluation framework partial (BPM complete, structure in progress)
- Annotator frontend components in development

See `openspec/` for active proposals and `openspec/archive/` for completed changes.
