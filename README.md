# EDM Track Analysis

[![CI](https://github.com/crsmithdev/edm/actions/workflows/ci.yml/badge.svg)](https://github.com/crsmithdev/edm/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A Python library and command-line tool for analyzing EDM tracks, providing BPM detection, structure analysis, and integration with external music services.

## Features

- **BPM Detection**: Accurate BPM detection using madmom and librosa
- **Structure Analysis**: Detect intro, buildup, drop, breakdown, and outro sections
- **External Service Integration**: Query Spotify, Beatport, and TuneBat for track information
- **Rich CLI**: Beautiful command-line interface with progress bars and tables
- **Configurable**: Support for configuration files and environment variables
- **Extensible**: Clean library API for programmatic use

## Installation

### Prerequisites

- Python 3.12 or higher
- ffmpeg (required for madmom audio file loading)
- System packages (Ubuntu/Debian):
   - python3-dev
   - build-essential

### Development Installation

1. **Install uv**      
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install madmom from source**:
   ```bash
   uv pip install --reinstall --no-cache "madmom @ git+https://github.com/CPJKU/madmom.git"
   ```
   
   Note: This step is necessary to avoid a wheel caching issue that causes incomplete builds of madmom.

4. **Verify installation**:
   ```bash
   uv run edm --version
   ```
## Quick start

Print usage information
```bash
uv run edm --help
```

Analyze an audio file:
```bash
uv run edm analyze path/to/audio/file.wav
```

## Configuration

### Spotify API Credentials

To enable Spotify BPM lookup, set your API credentials in a `.env` file in the project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Accuracy Evaluation

The project includes an internal accuracy evaluation framework for testing and validating analysis algorithms against reference data.

### Basic Usage

Evaluate BPM accuracy using CSV reference file:
```bash
uv run edm evaluate bpm --source ~/music --reference tests/fixtures/reference/bpm_tagged.csv
```

Evaluate using Spotify API (requires credentials):
```bash
uv run edm evaluate bpm --source ~/music --reference spotify
```

Evaluate using file metadata (ID3 tags):
```bash
uv run edm evaluate bpm --source ~/music --reference metadata
```

### Options

- `--source`: Directory containing audio files
- `--reference`: Reference source ('spotify', 'metadata', or path to CSV/JSON)
- `--sample-size N`: Number of files to sample (default: 100)
- `--full`: Evaluate all files
- `--seed N`: Random seed for reproducible sampling
- `--tolerance N`: BPM tolerance for accuracy (default: 2.5)
- `--output DIR`: Output directory for results

### Results

Results are saved to `benchmarks/results/accuracy/bpm/` in both JSON and Markdown formats:

- `latest.json` - Machine-readable full results
- `latest.md` - Human-readable summary
- Timestamped files with git commit hash for version tracking

See [benchmarks/results/README.md](benchmarks/results/README.md) for details.

## Development

- **Running Tests**:
   ```bash
   pytest
    ```

- **Running tests with coverage**:
   ```bash
   uv run pytest --cov=src --cov-report=term-missing
   ```

- **Dead Code Detection**:
   ```bash
   vulture src/ tests/ --min-confidence 60
   ```

- **Linting**
   ```bash
   uv run ruff check src/ tests/
   ```

- **Auto-Formatting**
   ```bash
   uv run black --check src/ tests/
   ```

- **Type Checking**
   ```bash
   uv run mypy src/ --ignore-missing-imports
   ```