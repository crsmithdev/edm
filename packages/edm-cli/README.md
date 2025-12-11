# EDM CLI

Command-line interface for analyzing Electronic Dance Music (EDM) tracks.

## Overview

`edm-cli` provides a Typer-based CLI built on top of the `edm-lib` core library. It offers commands for:

- **Audio Analysis**: Batch analyze BPM and structure for audio files
- **Model Training**: Train ML models for structure detection
- **Accuracy Evaluation**: Evaluate model performance against reference data
- **Data Management**: Import/export annotations in various formats
- **Model Management**: List and inspect trained models

## Installation

This package is part of the EDM monorepo. Install from the repository root:

```bash
# Install all packages
uv sync

# Verify installation
uv run edm --version
```

The `edm` command is automatically available after installation via the `[project.scripts]` entry point.

## Quick Start

### Analyze Audio Files

```bash
# Single file
uv run edm analyze track.mp3

# Multiple files
uv run edm analyze *.mp3

# Save to JSON
uv run edm analyze *.mp3 --output results.json

# Parallel processing (default: CPU count - 1)
uv run edm analyze *.mp3 --workers 8
```

### Train Models

```bash
# Quick training (10 epochs, small batch)
uv run edm train data/annotations \
    --audio-dir ~/music \
    --epochs 10 \
    --batch-size 2

# Standard training
uv run edm train data/annotations \
    --audio-dir ~/music \
    --epochs 50 \
    --batch-size 4 \
    --backbone mert-95m

# Full training with all options
uv run edm train data/annotations \
    --audio-dir ~/music \
    --output outputs/training \
    --epochs 100 \
    --batch-size 8 \
    --backbone mert-330m \
    --learning-rate 1e-4 \
    --boundary-head \
    --beat-head \
    --energy-head
```

### Evaluate Models

```bash
# Evaluate BPM detection
uv run edm evaluate bpm \
    --source ~/music \
    --reference metadata

# Evaluate structure detection
uv run edm evaluate structure \
    --source ~/music \
    --reference data/annotations \
    --output results.json
```

## Commands

### `edm analyze`

Analyze audio files for BPM and structure.

**Usage**:
```bash
edm analyze [OPTIONS] FILES...
```

**Options**:
- `--types TEXT` - Analysis types (comma-separated): `bpm`, `structure`, `beats`
- `--output PATH` - Output file (JSON/YAML)
- `--workers INT` - Number of parallel workers (default: CPU count - 1)
- `--no-metadata` - Force BPM computation (skip metadata)
- `--structure-detector TEXT` - Structure detector: `msaf`, `energy`
- `--cache-size INT` - Audio cache size (default: 10)
- `--recursive` - Recursively search directories
- `--log-level TEXT` - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`

**Examples**:
```bash
# BPM only
edm analyze *.mp3 --types bpm

# Structure only
edm analyze *.mp3 --types structure

# Both, save to JSON
edm analyze *.mp3 --types bpm,structure --output results.json

# Recursive analysis
edm analyze ~/music --recursive --output all_tracks.json

# Force computation (no metadata)
edm analyze *.mp3 --no-metadata
```

### `edm train`

Train ML models for structure detection.

**Usage**:
```bash
edm train [OPTIONS] ANNOTATION_DIR
```

**Options**:

**Required**:
- `ANNOTATION_DIR` - Directory containing YAML annotations

**Paths**:
- `--audio-dir PATH` - Audio file directory (default: `~/music`)
- `--output PATH` - Output directory (default: `outputs/training`)

**Model Architecture**:
- `--backbone TEXT` - Backbone model: `mert-95m`, `mert-330m`, `cnn` (default: `mert-95m`)
- `--freeze-backbone` - Freeze backbone weights (default: True)
- `--boundary-head` - Enable boundary detection head (default: True)
- `--beat-head` - Enable beat detection head (default: True)
- `--energy-head` - Enable energy prediction head (default: True)
- `--label-head` - Enable label classification head (default: False)

**Training Hyperparameters**:
- `--epochs INT` - Number of training epochs (default: 50)
- `--batch-size INT` - Batch size (default: 4)
- `--learning-rate FLOAT` - Learning rate (default: 1e-4)
- `--duration FLOAT` - Audio chunk duration in seconds (default: 30.0)

**Experiment Tracking**:
- `--experiment-name TEXT` - MLflow experiment name
- `--run-name TEXT` - MLflow run name

**Examples**:
```bash
# Quick training
edm train data/annotations --epochs 10 --batch-size 2

# Standard configuration
edm train data/annotations \
    --audio-dir ~/music \
    --epochs 50 \
    --batch-size 4 \
    --backbone mert-95m

# Large model, full training
edm train data/annotations \
    --backbone mert-330m \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4

# Custom experiment name
edm train data/annotations \
    --experiment-name "structure-detection-v2" \
    --run-name "mert-95m-50epochs"
```

### `edm evaluate`

Evaluate model accuracy against reference data.

**Usage**:
```bash
edm evaluate [SUBCOMMAND] [OPTIONS]
```

**Subcommands**:
- `bpm` - Evaluate BPM detection accuracy
- `structure` - Evaluate structure detection accuracy

**BPM Evaluation Options**:
- `--source PATH` - Audio directory
- `--reference TEXT` - Reference source: `metadata`, `csv`, `json`
- `--output PATH` - Output file for results

**Structure Evaluation Options**:
- `--source PATH` - Audio directory
- `--reference PATH` - Reference annotation directory
- `--tolerance FLOAT` - Boundary tolerance in seconds (default: 2.0)
- `--output PATH` - Output file for results

**Examples**:
```bash
# Evaluate BPM against metadata
edm evaluate bpm --source ~/music --reference metadata

# Evaluate structure with 3s tolerance
edm evaluate structure \
    --source ~/music \
    --reference data/annotations \
    --tolerance 3.0 \
    --output eval_results.json
```

### `edm data`

Data management and format conversion.

**Subcommands**:
- `import-rekordbox FILE` - Import Rekordbox XML
- `export-jams DIR` - Export annotations to JAMS format
- `validate DIR` - Validate annotation files

**Examples**:
```bash
# Import Rekordbox library
edm data import-rekordbox ~/music/rekordbox.xml

# Export to JAMS format
edm data export-jams data/annotations --output jams/

# Validate annotations
edm data validate data/annotations
```

### `edm models`

Model management and inspection.

**Subcommands**:
- `list` - List trained models
- `show NAME` - Show model details

**Examples**:
```bash
# List all models
edm models list

# Show specific model
edm models show mert-95m-boundary-v1
```

## Output Formats

### Terminal Output (Default)

Rich-formatted tables:

```
┌────────────────────┬───────┬────────────┐
│ File               │   BPM │ Confidence │
├────────────────────┼───────┼────────────┤
│ track1.mp3        │ 128.0 │       0.95 │
│ track2.mp3        │ 140.0 │       0.87 │
└────────────────────┴───────┴────────────┘
```

### JSON Output

```bash
edm analyze *.mp3 --output results.json
```

```json
[
  {
    "file": "track1.mp3",
    "duration": 240.5,
    "bpm": 128.0,
    "confidence": 0.95,
    "structure": [
      {"label": "intro", "start_time": 0.0, "end_time": 30.5},
      {"label": "buildup", "start_time": 30.5, "end_time": 60.0}
    ]
  }
]
```

### YAML Output

```bash
edm analyze track.mp3 --output track.yaml
```

```yaml
file: track.mp3
duration: 240.5
bpm: 128.0
confidence: 0.95
structure:
  - label: intro
    start_time: 0.0
    end_time: 30.5
```

## Performance

### Parallel Processing

By default, analysis uses `CPU_count - 1` workers for optimal throughput:

| Workers | Files | Time | Speedup |
|---------|-------|------|---------|
| 1 | 50 | ~20 min | 1x |
| 4 | 50 | ~5 min | 4x |
| 8 | 50 | ~3 min | 6-7x |

Control with `--workers` flag:

```bash
# Single-threaded
edm analyze *.mp3 --workers 1

# Specific worker count
edm analyze *.mp3 --workers 4

# Use all cores
edm analyze *.mp3 --workers $(nproc)
```

### Memory Usage

- **Per worker**: ~200MB (audio file in memory)
- **Cache**: ~2GB (10 tracks, adjustable with `--cache-size`)
- **Total (8 workers)**: ~3-4GB typical

Reduce workers if memory constrained:

```bash
edm analyze *.mp3 --workers 2 --cache-size 5
```

## Configuration

### Environment Variables

- `AUDIO_DIR` - Default audio directory (default: `~/music`)
- `ANNOTATION_DIR` - Annotation storage (default: `data/annotations`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

### Logging

Control log output:

```bash
# Debug logging
edm analyze track.mp3 --log-level DEBUG

# Save logs to file
edm analyze track.mp3 --log-file analysis.log

# Disable colored output
edm analyze track.mp3 --no-color
```

## Integration

### Python Scripts

Import and use programmatically:

```python
from edm_cli.commands.analyze import analyze_files
from pathlib import Path

files = list(Path("~/music").glob("*.mp3"))
results = analyze_files(files, workers=4)

for result in results:
    print(f"{result.file}: {result.bpm} BPM")
```

### Shell Scripts

```bash
#!/bin/bash
# Batch analyze multiple directories

for dir in ~/music/*/; do
    echo "Analyzing $dir"
    edm analyze "$dir" --recursive --output "$dir/analysis.json"
done
```

## Development

### Running from Source

```bash
# From repository root
uv run edm --help

# With local edits
uv run python -m edm_cli.main analyze track.mp3
```

### Testing

```bash
# Run CLI tests
uv run pytest packages/edm-cli/tests/
```

### Adding New Commands

1. Create command file in `src/edm_cli/commands/`
2. Import in `src/edm_cli/main.py`
3. Register with Typer app
4. Add tests

See [development.md](../../docs/development.md) for details.

## Documentation

- [CLI Reference](../../docs/cli-reference.md) - Complete command documentation
- [Architecture](../../docs/architecture.md) - System design
- [Development](../../docs/development.md) - Setup and testing

## Dependencies

This package depends on:
- `edm` (edm-lib) - Core analysis library
- `typer[all] >= 0.9.0` - CLI framework
- `rich >= 13.0.0` - Terminal formatting

All dependencies are managed via the monorepo workspace.

## License

See repository root for license information.

## Contributing

This package is part of the EDM monorepo. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development workflow.
