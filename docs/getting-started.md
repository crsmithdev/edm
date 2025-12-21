# Getting Started with EDM

Quick start guide for new EDM users.

## What is EDM?

EDM is a Python library and CLI tool for analyzing electronic dance music tracks:

- **BPM Detection**: Accurate tempo detection using neural networks
- **Structure Analysis**: Identify sections (intro, buildup, drop, breakdown, outro)
- **Training**: Train custom models on your own annotated data
- **Annotation Tool**: Web-based tool for creating training data

## Installation

### Prerequisites

- Python 3.12+
- ffmpeg
- System packages (Ubuntu/Debian): `python3-dev`, `build-essential`

### Quick Install

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/crsmithdev/edm.git
cd edm

# Install dependencies
uv sync

# Verify installation
uv run edm --version
```

## First Steps

### 1. Analyze Your First Track

```bash
# Analyze a single track
uv run edm analyze track.mp3

# Output shows BPM and structure
```

**Example output**:

```
Track: track.mp3
BPM: 128.0 (confidence: 0.95)

Structure:
  intro      0.0s - 32.0s  (bars 1-17)
  buildup    32.0s - 48.0s (bars 17-25)
  drop       48.0s - 96.0s (bars 25-49)
  breakdown  96.0s - 144.0s (bars 49-81)
  outro      144.0s - 180.0s (bars 81-97)
```

### 2. Batch Analysis

```bash
# Analyze all MP3 files in directory
uv run edm analyze ~/music/*.mp3 --output results.json

# Recursive directory analysis
uv run edm analyze ~/music/ --recursive
```

### 3. Try the Annotator

Create your own annotations with the web-based tool:

```bash
# Start annotator
just annotator

# Or manually start both servers:
# Terminal 1: Backend
cd packages/edm-annotator
uv run edm-annotator --env development

# Terminal 2: Frontend
cd packages/edm-annotator/frontend
pnpm run dev

# Open http://localhost:5174 in browser
```

For complete annotator guide, see [Annotator Guide](guides/annotator.md).

## Common Use Cases

### DJ / Producer

**Find BPM for mixing**:

```bash
# Quick BPM check
uv run edm analyze track.mp3 --types bpm

# Output: BPM: 128.0
```

**Identify drop timing**:

```bash
# Get structure
uv run edm analyze track.mp3 --types structure

# Find when the drop hits (e.g., bar 25)
```

### Researcher / ML Engineer

**Train custom models**:

```bash
# 1. Create annotations (use annotator tool)
just annotator

# 2. Train model
uv run edm train data/annotations --audio-dir ~/music --epochs 50

# 3. Evaluate
tensorboard --logdir outputs/training/
```

For complete training guide, see [Training Guide](guides/training.md).

### Data Scientist

**Analyze music collection**:

```bash
# Analyze all tracks
uv run edm analyze ~/music/ --recursive --output analysis.json

# Import into pandas
python -c "
import json
import pandas as pd

with open('analysis.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['tracks'])
print(df[['file', 'bpm', 'structure']].head())
"
```

## Configuration

### Environment Variables

```bash
# Set audio directory
export EDM_AUDIO_DIR=~/music

# Set annotation directory
export EDM_ANNOTATION_DIR=data/annotations

# Set log level
export EDM_LOG_LEVEL=INFO
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

### Configuration File

(Note: Config file loading not yet implemented - use CLI flags)

```bash
# Use CLI flags instead
uv run edm analyze track.mp3 --log-level DEBUG
```

## Understanding the Output

### BPM Detection

```
BPM: 128.0 (confidence: 0.95)
```

- **BPM**: Tempo in beats per minute
- **Confidence**: 0.0-1.0, higher is better
- **Source**: metadata (from ID3 tags) or computed (neural network)

### Structure Sections

```
intro      0.0s - 32.0s  (bars 1-17)
```

- **Label**: Section type (intro, buildup, drop, breakdown, outro)
- **Time**: Start and end in seconds
- **Bars**: Musical bar positions (1-indexed)

For complete terminology, see [Terminology Guide](reference/terminology.md).

## Next Steps

### Learn More

- **[CLI Reference](reference/cli.md)** - All command-line options
- **[Training Guide](guides/training.md)** - Train custom models
- **[Annotator Guide](guides/annotator.md)** - Create annotations
- **[Architecture](reference/architecture.md)** - How EDM works

### Get Help

- **Documentation Index**: [docs/INDEX.md](INDEX.md)
- **Troubleshooting**: [Troubleshooting Guide](reference/troubleshooting.md)
- **Issues**: [GitHub Issues](https://github.com/crsmithdev/edm/issues)

### Contribute

- **Development Setup**: [Development/Setup](development/setup.md)
- **Code Style**: [Python](development/code-style-python.md) | [JavaScript](development/code-style-javascript.md)
- **Testing**: [Testing Guide](development/testing.md)

## Quick Reference

### Essential Commands

```bash
# Analysis
uv run edm analyze track.mp3                    # Single track
uv run edm analyze ~/music/ --recursive         # Directory

# Annotator
just annotator                                   # Start web tool

# Training
uv run edm train data/annotations --audio-dir ~/music

# Testing
uv run pytest                                    # Run tests

# Code quality
uv run ruff check --fix .                       # Lint and fix
```

For complete command reference, see [Common Commands Cheatsheet](cheatsheets/common-commands.md).

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` from repo root |
| `FileNotFoundError: audio` | Check file path or set `EDM_AUDIO_DIR` |
| `command not found: uv` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Tests failing | Run `uv sync` and retry |

For complete troubleshooting, see [Troubleshooting Guide](reference/troubleshooting.md).

## What's Next?

**Choose your path**:

- **Just want to analyze tracks?** → See [CLI Reference](reference/cli.md)
- **Want to create annotations?** → See [Annotator Guide](guides/annotator.md)
- **Want to train models?** → See [Training Guide](guides/training.md)
- **Want to contribute?** → See [Development Setup](development/setup.md)

---

**Questions?** Check the [Documentation Index](INDEX.md) or [open an issue](https://github.com/crsmithdev/edm/issues).
