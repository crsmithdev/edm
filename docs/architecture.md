# Architecture

## Overview

EDM is a Python library and CLI for analyzing EDM tracks. It provides BPM detection, structure analysis, and integration with external music services (Spotify, Beatport, TuneBat).

## Module Organization

```
src/
├── cli/                    # CLI entry point
│   └── main.py            # Typer app with analyze/evaluate commands
└── edm/                   # Core library
    ├── analysis/          # Audio analysis
    │   ├── bpm.py         # Cascading BPM strategy (metadata → spotify → computed)
    │   ├── bpm_detector.py # BPM computation (beat_this, librosa)
    │   └── structure.py   # Structure detection (intro, drop, etc.) [PLACEHOLDER]
    ├── evaluation/        # Accuracy evaluation framework
    │   ├── common.py      # Shared utilities, metrics (MAE, RMSE)
    │   ├── reference.py   # Reference data sources
    │   └── evaluators/    # Evaluation implementations
    │       └── bpm.py     # BPM accuracy evaluation
    ├── external/          # External service integrations
    │   ├── spotify.py     # Spotify API client
    │   ├── beatport.py    # Beatport client
    │   └── tunebat.py     # TuneBat client
    ├── features/          # Feature extraction
    │   ├── temporal.py    # Time-domain features
    │   └── spectral.py    # Frequency-domain features
    ├── io/                # File I/O
    │   └── metadata.py    # Audio file metadata (ID3, FLAC, etc.)
    ├── models/            # Data models
    │   └── base.py        # Pydantic models
    ├── config.py          # Configuration (Pydantic, env vars)
    ├── exceptions.py      # Custom exceptions
    └── logging.py         # Structlog configuration
```

## Key Components

### BPM Detection (`src/edm/analysis/bpm.py:40`)

Uses a cascading lookup strategy:

1. **Metadata** (fastest) - Read BPM from ID3/FLAC/MP4 tags
2. **Spotify** (professional accuracy) - Query Spotify API using artist/title
3. **Computed** (fallback) - Analyze audio with beat_this or librosa

Control via CLI flags:
- `--offline`: Skip Spotify (metadata → computed)
- `--ignore-metadata`: Skip metadata (spotify → computed)
- Both flags: Force computation only

### BPM Computation (`src/edm/analysis/bpm_detector.py:224`)

Two methods available:
- **beat_this** (default): Neural network beat tracker (ISMIR 2024), high accuracy for EDM
- **librosa**: Standard tempo detection, used as fallback

Both methods:
- Handle tempo multiplicity (half/double time detection)
- Adjust to preferred EDM range (120-150 BPM)
- Return confidence scores based on beat interval consistency

### External Services (`src/edm/external/`)

Pattern for service integration:
1. Read track metadata (artist, title)
2. Query external API
3. Return structured result with BPM/confidence
4. Handle authentication via environment variables

Configuration in `.env`:
```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

### Evaluation Framework (`src/edm/evaluation/`)

Tests analysis accuracy against reference data:
- **Reference sources**: Spotify API, file metadata, CSV/JSON files
- **Metrics**: MAE, RMSE, accuracy within tolerance
- **Output**: JSON (machine-readable) + Markdown (human-readable)
- **Sampling**: Random subset with optional seed for reproducibility

Results saved to `benchmarks/results/accuracy/bpm/` with `latest.*` symlinks.

### Logging (`src/edm/logging.py:26`)

Uses structlog with:
- Console output (colored, human-readable) for development
- JSON output for production/file logging
- Context variables for request tracing
- Configurable via CLI flags (`--log-level`, `--json-logs`, `--log-file`)

### Configuration (`src/edm/config.py`)

Pydantic models with:
- Environment variable support (`SPOTIFY_CLIENT_ID`, etc.)
- TOML config file support (planned)
- Nested configs: `AnalysisConfig`, `ExternalServicesConfig`

## Data Flow

### Analysis Flow

```
Audio File → Metadata Read → BPM Strategy → Result
                ↓
            (if needed)
                ↓
            Spotify API
                ↓
            (if needed)
                ↓
            Audio Load → madmom/librosa → BPM Computation
```

### Evaluation Flow

```
Source Directory → File Discovery → Sampling → Analysis
        ↓                                         ↓
Reference Source → Reference Values         Computed Values
        ↓                                         ↓
                    Error Calculation
                          ↓
                    Metrics (MAE, RMSE)
                          ↓
                    Results (JSON, MD)
```

## Design Decisions

### Why beat_this for BPM?

- Neural network beat tracker from ISMIR 2024
- Designed specifically for accurate beat tracking in music
- Outperforms madmom and librosa for EDM detection
- Trade-off: Slower, resource-intensive during initialization

### Why cascading BPM strategy?

- Metadata is instant when available
- Spotify provides professional-grade BPM values
- Computation is resource-intensive, used only when needed
- User can control strategy via CLI flags

### Why structlog?

- Structured logging enables log aggregation
- Context binding tracks operations across calls
- Dual output (console + JSON file) supports both dev and prod
- Better performance than standard logging

### Why Pydantic for config?

- Type validation catches config errors early
- Environment variable support built-in
- Clear schema documentation via type hints
- Easy serialization/deserialization

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| CLI | typer + rich | Modern CLI with colors/tables |
| Audio Analysis | beat_this, librosa | BPM detection, audio loading |
| Audio I/O | mutagen | Metadata reading |
| HTTP Client | requests, spotipy | External API calls |
| Config | pydantic | Validation, env vars |
| Logging | structlog | Structured logging |
| Testing | pytest | Test framework |
| Linting | ruff | Fast linting |
| Type Checking | mypy | Static analysis |

## Placeholder / Unimplemented Features

The following features are documented but currently return hardcoded/placeholder values:

### Structure Analysis (`src/edm/analysis/structure.py`)

**Status:** Placeholder implementation only

- Always returns the same hardcoded sections (intro, buildup, drop)
- Does not analyze actual audio content
- Returns actual audio duration (since fix) but sections are static
- **TODO:** Implement actual structure detection algorithm

### External Service Integrations

#### Beatport (`src/edm/external/beatport.py`)

**Status:** Not implemented

- Always returns `None`
- **TODO:** Implement Beatport API integration or web scraper

#### TuneBat (`src/edm/external/tunebat.py`)

**Status:** Not implemented

- Always returns `None`
- **TODO:** Implement TuneBat API integration or web scraper

### Configuration File Support (`src/edm/config.py`)

**Status:** Partial implementation

- TOML configuration file path is recognized and logged
- File loading and parsing is not yet implemented
- Code currently returns default configuration regardless of file contents
- **TODO:** Complete TOML parsing and configuration loading
