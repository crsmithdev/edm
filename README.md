# EDM Track Analysis

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

- Python 3.9 or higher (tested with Python 3.12)
- pip or pip3
- git
- ffmpeg (required for madmom audio file loading)
- System packages (Ubuntu/Debian):
  ```bash
  sudo apt update
  sudo apt install -y python3-pip python3-venv python3-dev build-essential ffmpeg
  ```

### From Source (Recommended for Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/crsmithdev/edm.git
   cd edm
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   
   First, install Cython and NumPy (required for building madmom):
   ```bash
   pip install Cython numpy
   ```
   
   Then install core dependencies:
   ```bash
   pip install librosa pydantic "typer[all]" rich spotipy requests mutagen
   ```
   
   Install madmom from source (required for accurate BPM detection):
   ```bash
   pip install git+https://github.com/CPJKU/madmom.git
   ```

4. **Install the edm package in editable mode**:
   ```bash
   pip install --no-deps -e .
   ```

5. **Verify installation**:
   ```bash
   edm --version
   ```

### Alternative: Install with pip (when dependencies resolve)

Once all dependencies are available on PyPI for your Python version:

```bash
pip install -e .
```

### Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

### Why Install madmom from Source?

The PyPI version of madmom (0.16.1) has build issues with Python 3.12 due to Cython dependencies. Installing from the GitHub repository (version 0.17.dev0) provides:
- Python 3.12 compatibility
- Latest bug fixes and improvements
- Better performance on modern systems

If you encounter issues building madmom, you can use librosa-only mode by passing `use_madmom=False` to the analysis functions, though madmom provides more accurate BPM detection for EDM tracks.

## Quick Start

### Command Line

Analyze a single track:
```bash
edm analyze track.mp3
```

Analyze with specific types:
```bash
edm analyze track.mp3 --types bpm,grid
```

Analyze multiple tracks:
```bash
edm analyze *.mp3
```

Analyze directory recursively:
```bash
edm analyze /path/to/tracks/ --recursive
```

Save results to JSON:
```bash
edm analyze track.mp3 --output results.json
```

Enable verbose logging:
```bash
edm analyze track.mp3 --verbose
```

### Python Library

```python
from pathlib import Path
from edm.analysis import analyze_bpm, analyze_structure

# Analyze BPM
result = analyze_bpm(Path("track.mp3"))
print(f"BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")

# Analyze structure
structure = analyze_structure(Path("track.mp3"))
for section in structure.sections:
    print(f"{section.label}: {section.start_time:.1f}s - {section.end_time:.1f}s")
```

## Configuration

### Spotify API Credentials

To enable Spotify BPM lookup, set your API credentials in a `.env` file in the project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### BPM Lookup Strategy

EDM uses a cascading strategy for BPM detection:

1. **Metadata** (fastest): Read BPM from file tags (ID3v2, MP4, FLAC)
2. **Spotify API** (accurate): Professional BPM data from Spotify
3. **Computed** (most accurate for EDM): madmom DBN beat tracking

Control the strategy with flags:
- `--ignore-metadata`: Skip file metadata (uses Spotify → computed)
- `--offline`: Skip Spotify API (uses metadata → computed)
- `--offline --ignore-metadata`: Force computation only

### Configuration File

Create `~/.config/edm/config.toml`:

```toml
log_level = "INFO"

[analysis]
detect_bpm = true
detect_structure = true
use_madmom = true
use_librosa = false

[external_services]
enable_beatport = true
enable_tunebat = true
cache_ttl = 3600
```

### Environment Variables

Alternative to `.env` file:

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
export EDM_LOG_LEVEL=DEBUG
```

## CLI Reference

### Global Options

- `--version`, `-v`: Show version and exit
- `--help`: Show help message

### `analyze` Command

Analyze EDM tracks for BPM, structure, and other features.

**Arguments:**
- `FILES`: Audio files to analyze (required)

**Options:**
- `--types`, `-t TEXT`: Comma-separated analysis types (bpm,grid,structure)
- `--output`, `-o PATH`: Save results to JSON file
- `--format`, `-f TEXT`: Output format (table, json) [default: table]
- `--config`, `-c PATH`: Path to configuration file
- `--recursive`, `-r`: Recursively analyze directories
- `--verbose`: Enable verbose logging (DEBUG level)
- `--quiet`, `-q`: Suppress non-essential output
- `--no-color`: Disable colored output

**Examples:**
```bash
# Basic analysis
edm analyze track.mp3

# BPM only
edm analyze track.mp3 --types bpm

# Multiple types
edm analyze track.mp3 --types bpm,grid

# Batch analysis
edm analyze *.mp3 --output results.json

# Recursive directory
edm analyze /music/edm/ --recursive --format json

# Verbose mode
edm analyze track.mp3 --verbose
```

## Architecture

### Library Structure

```
src/edm/
├── __init__.py           # Package entry point
├── analysis/             # Analysis algorithms
│   ├── bpm.py           # BPM detection
│   └── structure.py     # Structure analysis
├── io/                   # File I/O
│   ├── audio.py         # Audio loading
│   └── metadata.py      # Metadata reading
├── external/             # External API clients
│   ├── spotify.py       # Spotify integration
│   ├── beatport.py      # Beatport integration
│   └── tunebat.py       # TuneBat integration
├── features/             # Feature extraction
│   ├── spectral.py      # Spectral features
│   └── temporal.py      # Temporal features
├── models/               # ML models
│   └── base.py          # Model loading
├── config.py             # Configuration management
└── exceptions.py         # Custom exceptions
```

### CLI Structure

```
src/cli/
├── __init__.py           # CLI package
├── main.py               # Entry point with Typer
└── commands/             # Command implementations
    └── analyze.py       # Analyze command
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
```

### Linting

```bash
ruff src tests
```

### Type Checking

```bash
mypy src
```

### Dead Code Detection

Check for unused code:

```bash
vulture src/ tests/ --min-confidence 60
```

Higher confidence levels (80-100) reduce false positives but may miss some dead code.

## Dependencies

### Core Libraries
- **librosa**: Audio processing and analysis
- **madmom**: BPM detection and beat tracking
- **essentia**: Music information retrieval
- **numpy**: Numerical computing

### CLI & Configuration
- **typer**: Modern CLI framework
- **rich**: Beautiful terminal output
- **pydantic**: Configuration validation

### External Services
- **spotipy**: Spotify API client
- **requests**: HTTP client

## Logging

Logs are written to `~/.local/share/edm/logs/edm.log` by default.

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages

### Logging vs CLI Output

- **CLI output** (stdout): User-facing information, progress, results
- **Logs** (file): Detailed diagnostic information for debugging

The library never uses `print()` statements. All output goes through either Rich (CLI) or logging (diagnostics).

## License

[Your License Here]

## Contributing

Contributions are welcome! Please open an issue or pull request.

## Credits

Built with:
- [librosa](https://librosa.org/)
- [madmom](https://madmom.readthedocs.io/)
- [essentia](https://essentia.upf.edu/)
- [Typer](https://typer.tiangolo.com/)
- [Rich](https://rich.readthedocs.io/)