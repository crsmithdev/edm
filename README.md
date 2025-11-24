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

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Verify installation**:
   ```bash
   edm --version
   ```
## Quick start

Print usage information
```bash
edm --help
```
## Configuration

### Spotify API Credentials

To enable Spotify BPM lookup, set your API credentials in a `.env` file in the project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

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