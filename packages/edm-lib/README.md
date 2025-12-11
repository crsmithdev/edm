# EDM Library

Core Python library for Electronic Dance Music (EDM) track analysis, ML model training, and evaluation.

## Overview

`edm-lib` provides:

- **Audio Analysis**: BPM detection, structure analysis, beat tracking, and bar calculations
- **ML Training**: Multi-task learning for structure detection with MERT and CNN backbones
- **Model Evaluation**: Accuracy metrics for BPM and structure predictions
- **Data Management**: Schema validation, format conversion (JAMS, Rekordbox XML), and annotation I/O
- **Processing**: Parallel audio analysis with LRU caching

This library powers the `edm-cli` command-line interface and can be used programmatically in Python applications.

## Installation

This package is part of the EDM monorepo. Install from the repository root:

```bash
# Install all packages in workspace
uv sync

# Or install just edm-lib
uv pip install -e packages/edm-lib
```

## Quick Start

### Analyze Audio

```python
from edm.analysis import analyze_bpm, analyze_structure
from pathlib import Path

# Analyze BPM
bpm_result = analyze_bpm(Path("track.mp3"))
print(f"BPM: {bpm_result.bpm} (confidence: {bpm_result.confidence})")

# Analyze structure
structure_result = analyze_structure(Path("track.mp3"))
for section in structure_result.sections:
    print(f"{section.label}: {section.start_time:.2f}s - {section.end_time:.2f}s")
```

### Train Models

```python
from edm.training import Trainer, TrainingConfig
from pathlib import Path

config = TrainingConfig(
    annotation_dir=Path("data/annotations"),
    audio_dir=Path("~/music").expanduser(),
    epochs=50,
    batch_size=4,
    backbone="mert-95m",
)

trainer = Trainer(config)
trainer.train()
```

### Load Annotations

```python
from edm.data.schema import Annotation
from pathlib import Path

annotation = Annotation.from_yaml(Path("data/annotations/track.yaml"))

print(f"BPM: {annotation.audio.bpm}")
print(f"Downbeat: {annotation.audio.downbeat}s")

for section in annotation.structure:
    print(f"Bar {section.bar}: {section.label} at {section.time}s")
```

## Architecture

### Two-Tier Design

Analysis modules separate public APIs from implementation details:

- **Tier 1: Public APIs** (`bpm.py`, `structure.py`) - Stable interfaces
- **Tier 2: Detectors** (`*_detector.py`) - Algorithm implementations

Example:
```python
# Use public API (recommended)
from edm.analysis import analyze_bpm

# Detectors are implementation details (internal use)
from edm.analysis.bpm_detector import compute_bpm  # Not recommended for external use
```

### Cascading Strategies

BPM detection uses multiple methods with intelligent fallback:

1. **Metadata** (~1ms) - Read from file tags
2. **beat_this** (~10-15s) - Neural network beat tracker
3. **librosa** (~5-8s) - Traditional spectral analysis

Structure detection similarly cascades:
1. **MSAF** - Spectral flux segmentation
2. **Energy-based** - RMS + spectral contrast fallback

## Module Organization

```
src/edm/
├── analysis/         Audio analysis (BPM, structure, beats, bars)
├── training/         ML model training and dataset loading
├── models/           Neural network architectures (backbones, heads)
├── data/             Schema, validation, format conversion
├── evaluation/       Accuracy metrics and evaluators
├── io/               Audio loading, metadata extraction
├── processing/       Parallel execution utilities
├── registry/         MLflow model registry integration
├── config.py         Configuration management
├── exceptions.py     Custom exception classes
└── logging.py        Structured logging setup
```

See [docs/architecture.md](../../docs/architecture.md) for complete system design.

## Key Features

### BPM Detection

Cascading strategy for speed vs. accuracy:

```python
from edm.analysis import analyze_bpm
from pathlib import Path

# Auto-cascades: metadata → beat_this → librosa
result = analyze_bpm(Path("track.mp3"))

# Force computation (skip metadata)
result = analyze_bpm(Path("track.mp3"), use_metadata=False)

# Access alternatives
print(f"Primary: {result.bpm}")
print(f"Alternatives: {result.alternatives}")
```

### Structure Detection

5 EDM section types with bar-aligned boundaries:

```python
from edm.analysis import analyze_structure
from pathlib import Path

result = analyze_structure(Path("track.mp3"))

for section in result.sections:
    print(f"{section.label:10s} | "
          f"Bar {section.start_bar:5.1f}-{section.end_bar:5.1f} | "
          f"{section.start_time:6.2f}s-{section.end_time:6.2f}s")

# Output:
# intro      | Bar   1.0- 33.0 |   0.00s- 62.11s
# buildup    | Bar  33.0- 49.0 |  62.11s- 92.58s
# drop       | Bar  49.0- 81.0 |  92.58s-153.05s
# breakdown  | Bar  81.0- 97.0 | 153.05s-183.52s
# outro      | Bar  97.0-113.0 | 183.52s-213.99s
```

### Model Training

Multi-task learning with MERT or CNN backbones:

```python
from edm.training import Trainer, TrainingConfig
from pathlib import Path

config = TrainingConfig(
    annotation_dir=Path("data/annotations"),
    audio_dir=Path("~/music").expanduser(),
    output_dir=Path("outputs/training"),

    # Model architecture
    backbone="mert-95m",  # or "mert-330m", "cnn"
    freeze_backbone=True,

    # Training hyperparameters
    epochs=50,
    batch_size=4,
    learning_rate=1e-4,

    # Task heads
    boundary_head=True,
    beat_head=True,
    energy_head=True,
    label_head=False,  # Less stable, optional

    # MLflow tracking
    experiment_name="edm-structure-detection",
)

trainer = Trainer(config)
metrics = trainer.train()

print(f"Final F1: {metrics['boundary_f1']:.3f}")
```

### Data Validation

Pydantic-based schemas with runtime validation:

```python
from edm.data.schema import Annotation, AudioMetadata, Structure
from pathlib import Path

# Load and validate
annotation = Annotation.from_yaml(Path("track.yaml"))

# Validation errors raise helpful messages
try:
    annotation = Annotation.from_yaml(Path("bad_track.yaml"))
except ValidationError as e:
    print(e)  # Clear error messages with field paths
```

### Parallel Processing

LRU-cached audio loading with multiprocessing:

```python
from edm.processing.parallel import parallel_analyze
from pathlib import Path

files = list(Path("~/music").glob("*.mp3"))

# Parallel analysis with worker pool
results = parallel_analyze(
    files,
    workers=8,          # Default: CPU count - 1
    cache_size=10,      # LRU cache for ~2GB memory
)

for file, result in zip(files, results):
    print(f"{file.name}: {result.bpm} BPM")
```

## Development

### Running Tests

```bash
# All tests
uv run pytest packages/edm-lib/tests/

# Unit tests only
uv run pytest packages/edm-lib/tests/unit/

# With coverage
uv run pytest packages/edm-lib/tests/ --cov=edm --cov-report=term
```

### Code Quality

```bash
# Lint
uv run ruff check packages/edm-lib/src/

# Format
uv run ruff format packages/edm-lib/src/

# Type check
uv run mypy packages/edm-lib/src/
```

## Documentation

- [Architecture](../../docs/architecture.md) - System design and module organization
- [Training Guide](../../docs/training.md) - Model training workflows
- [Development](../../docs/development.md) - Setup and testing
- [Agent Guide](../../docs/agent-guide.md) - Navigation for AI assistants

## Dependencies

### Core Analysis
- `librosa >= 0.10.0` - Audio processing
- `beat_this` - Neural beat tracking (ISMIR 2024)
- `msaf >= 0.1.80` - Music segmentation

### Deep Learning
- `torch >= 2.0.0` - Neural networks
- `torchaudio >= 2.0.0` - Audio transforms
- `transformers >= 4.30.0` - MERT models

### Data & Validation
- `pydantic >= 2.0.0` - Schema validation
- `pyyaml >= 6.0.0` - YAML I/O
- `jams >= 0.3.5` - MIR annotation format

### Experiment Tracking
- `mlflow >= 2.10.0` - Experiment tracking
- `tensorboard >= 2.0.0` - Visualization
- `dvc >= 3.0.0` - Data versioning

See `pyproject.toml` for complete dependency list.

## License

See repository root for license information.

## Contributing

This package is part of the EDM monorepo. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development workflow.
