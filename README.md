# EDM Track Analysis

[![CI](https://github.com/crsmithdev/edm/actions/workflows/ci.yml/badge.svg)](https://github.com/crsmithdev/edm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/crsmithdev/edm/graph/badge.svg)](https://codecov.io/gh/crsmithdev/edm)
[![GitHub release](https://img.shields.io/github/v/release/crsmithdev/edm)](https://github.com/crsmithdev/edm/releases)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A Python library and CLI for analyzing EDM tracks, providing BPM detection, structure analysis, and integration with external music services.

## Installation

### Prerequisites

- Python 3.12+
- ffmpeg
- System packages (Ubuntu/Debian): `python3-dev`, `build-essential`

### Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install madmom from source
uv pip install --reinstall --no-cache "madmom @ git+https://github.com/CPJKU/madmom.git"

# Verify
uv run edm --version
```

## Quick Start

```bash
# Analyze an audio file
uv run edm analyze track.mp3

# Analyze with specific types
uv run edm analyze track.mp3 --types bpm,structure

# Save results to JSON
uv run edm analyze *.mp3 --output results.json
```

## Performance

### Parallel Processing

By default, analysis uses all available CPU cores minus one for optimal performance. Control parallelism with the `--workers` flag:

```bash
# Use default (CPU count - 1)
uv run edm analyze *.mp3

# Use specific worker count
uv run edm analyze *.mp3 --workers 4

# Single-threaded processing
uv run edm analyze *.mp3 --workers 1
```

### Benchmarks

Typical performance on an 8-core system:

| Files | Workers | Time | Speedup |
|-------|---------|------|---------|
| 50 | 1 | ~20 min | 1x |
| 50 | 4 | ~5 min | 4x |
| 50 | 8 | ~3 min | 6-7x |

Performance scales with CPU cores. BPM detection is CPU-bound, so more workers = faster processing.

### Memory Requirements

Each worker loads one audio file into memory. Budget approximately:
- ~200MB per worker for typical audio files
- On 8-core system with 16GB RAM: 8 workers uses ~1.6GB
- Adjust `--workers` if running low on memory

## Configuration

Set Spotify API credentials in `.env`:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Documentation

- [CLI Reference](docs/cli-reference.md) - Complete command documentation
- [Architecture](docs/architecture.md) - System design and module organization
- [Development](docs/development.md) - Setup, testing, and code quality

## Development

```bash
uv run pytest -v                           # Run tests
uv run ruff check --fix . && ruff format . # Lint and format
uv run mypy src/                           # Type check
```

See [development.md](docs/development.md) for details.
