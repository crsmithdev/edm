# Architecture

## Overview

EDM is a Python library and CLI for analyzing EDM tracks. It provides BPM detection and structure analysis using neural network-based audio processing.

## Design Patterns

### Two-Tier Abstraction

Analysis modules use a two-tier pattern separating public API from detector implementations:

**Tier 1: Public API** (`bpm.py`, `structure.py`)
- Simple function interface: `analyze_bpm(filepath)`, `analyze_structure(filepath)`
- Strategy selection and fallback logic
- Result aggregation and formatting
- User-facing documentation

**Tier 2: Detectors** (`bpm_detector.py`, `structure_detector.py`)
- Algorithm implementations: beat_this, librosa, MSAF, energy
- Internal data models: `ComputedBPM`, `DetectedSection`
- No strategy logic (focused on single algorithm)
- Technical implementation details

**Benefits**:
- Public API stable across detector changes
- Easy to add new detectors without breaking callers
- Clear separation of concerns (strategy vs implementation)

**Example**:
```python
# Public API (bpm.py)
def analyze_bpm(filepath):
    # Try metadata first
    # Fall back to beat_this detector
    # Fall back to librosa detector
    return BPMResult(...)

# Detector (bpm_detector.py)
def compute_bpm_beat_this(filepath):
    # Pure beat_this implementation
    return ComputedBPM(...)
```

### Audio Caching Strategy

Audio loading is expensive (I/O + decoding). Cache implemented in `src/edm/io/audio.py`:

**LRU Cache**:
- OrderedDict-based LRU eviction
- Configurable size (default: 10 tracks)
- Cache key: `(filepath, sample_rate)`
- Thread-safe for future async support

**Usage**:
```python
# First call: loads from disk
audio1, sr = load_audio(path, sr=22050)  # cache miss

# Second call (same path + sr): returns cached
audio2, sr = load_audio(path, sr=22050)  # cache hit

# Different sample rate: new cache entry
audio3, sr = load_audio(path, sr=44100)  # cache miss (different key)
```

**Cache Control**:
- `set_cache_size(n)`: Change max cached files
- `clear_audio_cache()`: Flush all cached audio
- `cache_size=0`: Disable caching entirely

**Benefits**:
- Multiple analyses (BPM + structure) load audio once
- Evaluation loops reuse audio across reference comparisons
- Memory-bounded (doesn't grow indefinitely)

Implementation: `src/edm/io/audio.py:16`

## Module Organization

```
src/
├── cli/                    # CLI entry point
│   └── main.py            # Typer app with analyze/evaluate commands
└── edm/                   # Core library
    ├── analysis/          # Audio analysis
    │   ├── bpm.py         # Cascading BPM strategy (metadata → computed)
    │   ├── bpm_detector.py # BPM computation (beat_this, librosa)
    │   ├── structure.py   # Structure analysis public API
    │   └── structure_detector.py # Structure detectors (MSAF, energy)
    ├── evaluation/        # Accuracy evaluation framework
    │   ├── common.py      # Shared utilities, metrics (MAE, RMSE)
    │   ├── reference.py   # Reference data sources
    │   └── evaluators/    # Evaluation implementations
    │       ├── bpm.py     # BPM accuracy evaluation
    │       └── structure.py # Structure accuracy evaluation
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

### BPM Detection (`src/edm/analysis/bpm.py:42`)

Uses a cascading lookup strategy:

1. **Metadata** (fastest) - Read BPM from ID3/FLAC/MP4 tags
2. **Computed** (fallback) - Analyze audio with beat_this or librosa

Control via CLI flags:
- `--ignore-metadata`: Skip metadata (force computation)

### BPM Computation (`src/edm/analysis/bpm_detector.py:224`)

Two methods available:
- **beat_this** (default): Neural network beat tracker (ISMIR 2024), high accuracy for EDM
- **librosa**: Standard tempo detection, used as fallback

Both methods:
- Handle tempo multiplicity (half/double time detection)
- Adjust to preferred EDM range (120-150 BPM)
- Return confidence scores based on beat interval consistency

### Structure Detection (`src/edm/analysis/structure.py`)

Detects track structure sections using MSAF-based boundary detection with energy-based labeling:

**Detector Selection** (`--structure-detector`):
- `auto` (default): Use MSAF if available, fall back to energy
- `msaf`: MSAF boundary detection with energy-based EDM label mapping
- `energy`: Rule-based detection using RMS energy and spectral contrast

**Section Labels** (EDM terminology):
- `intro` - Opening section
- `buildup` - Rising energy, tension building
- `drop` - High energy payoff section
- `breakdown` - Reduced energy, melodic focus
- `outro` - Closing section

**MSAF Integration** (`src/edm/analysis/structure_detector.py:55`):
- Music Structure Analysis Framework (Nieto & Bello, ISMIR 2016)
- Boundary detection using spectral flux algorithm
- Energy-based mapping to EDM labels (high energy → drop, low energy → breakdown)
- Returns confidence scores based on energy characteristics

**Energy-Based Fallback** (`src/edm/analysis/structure_detector.py:260`):
- RMS energy analysis for drop detection
- Boundary detection via energy gradient peaks
- Ensures full track coverage with no gaps
- Minimum section duration filtering (8 seconds)

**Bar/Measure Calculation** (`src/edm/analysis/bars.py`):
- Converts time positions to musical bars based on BPM and time signature
- Automatically integrates with structure analysis (calculates bars for all sections)
- Designed for future beat grid integration (optional `beat_grid` parameter reserved)
- Default 4/4 time signature, supports 3/4, 6/8, etc.
- Graceful degradation: bar fields are None when BPM unavailable
- Utility functions: `time_to_bars()`, `bars_to_time()`, `bar_count_for_range()`

**Output Format:**
- Sections include both time (`start_time`, `end_time`) and bar positions (`start_bar`, `end_bar`, `bar_count`)
- JSON output: `{"label": "drop", "start": 30.0, "end": 90.0, "start_bar": 17.0, "end_bar": 49.0, "bar_count": 32.0}`
- Bar numbering is 1-indexed (bar 1 = first bar) to match DJ software conventions
- Example: "32 bars (30.0s-90.0s)" is more meaningful than "30.0s-90.0s" for musical analysis

### Evaluation Framework (`src/edm/evaluation/`)

Tests analysis accuracy against reference data:
- **Reference sources**: File metadata, CSV/JSON files
- **Metrics**: MAE, RMSE, accuracy within tolerance (BPM); Precision, Recall, F1 (structure)
- **Output**: JSON (machine-readable) + Markdown (human-readable)
- **Sampling**: Random subset with optional seed for reproducibility

**BPM Evaluation** (`src/edm/evaluation/evaluators/bpm.py`):
- Compares computed BPM against reference values
- Results saved to `data/accuracy/bpm/`

**Structure Evaluation** (`src/edm/evaluation/evaluators/structure.py`):
- Compares detected sections against ground truth annotations
- Supports both time-based and bar-based annotation formats
- Bar-based annotations automatically converted to time using BPM
- Boundary tolerance matching (default ±2 seconds)
- Per-section-type metrics (precision/recall/F1 for each label)
- Results saved to `data/accuracy/structure/`

### Logging (`src/edm/logging.py:26`)

Uses structlog with:
- Console output (colored, human-readable) for development
- JSON output for production/file logging
- Context variables for request tracing
- Configurable via CLI flags (`--log-level`, `--json-logs`, `--log-file`)

### Configuration (`src/edm/config.py`)

Pydantic models with:
- Environment variable support
- TOML config file support (planned)
- Nested configs: `AnalysisConfig`

## Module Dependency Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ CLI Layer (src/cli/)                                        │
│ ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│ │ main.py     │  │ analyze.py  │  │ evaluate.py  │         │
│ └──────┬──────┘  └──────┬──────┘  └──────┬───────┘         │
└────────┼─────────────────┼─────────────────┼────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Analysis Layer (src/edm/analysis/)                          │
│ ┌──────────┐   ┌─────────────────┐   ┌────────────┐        │
│ │ bpm.py   │   │ structure.py    │   │ bars.py    │        │
│ └────┬─────┘   └────┬────────────┘   └─────┬──────┘        │
│      │              │                       │               │
│      ▼              ▼                       │               │
│ ┌───────────────┐  ┌───────────────────┐   │               │
│ │bpm_detector.py│  │structure_detector │   │               │
│ └───────┬───────┘  └──────┬────────────┘   │               │
└─────────┼──────────────────┼────────────────┼───────────────┘
          │                  │                │
          ▼                  ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│ I/O Layer (src/edm/io/)                                     │
│ ┌──────────┐   ┌──────────┐                                │
│ │ audio.py │   │metadata  │                                │
│ │(caching) │   │  .py     │                                │
│ └──────────┘   └──────────┘                                │
└─────────────────────────────────────────────────────────────┘

External Dependencies:
  bpm_detector.py → beat_this, librosa
  structure_detector.py → msaf, librosa
  audio.py → librosa
  metadata.py → mutagen
```

**Key Dependencies**:
- Public APIs (`bpm.py`, `structure.py`) depend on detectors but not vice versa
- Detectors are independent (can be tested in isolation)
- All analysis depends on `io/audio.py` for audio loading
- CLI layer only imports public APIs (never detectors directly)

## Data Flow

### Analysis Flow

```
Audio File → Metadata Read → BPM Strategy → Result
                ↓
            (if needed)
                ↓
            Audio Load (cached) → beat_this/librosa → BPM Computation
                ↑
                └─ LRU Cache (10 tracks)
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
- Computation is resource-intensive, used only when needed
- User can control strategy via CLI flags

### Why no external APIs?

- External BPM APIs are unreliable (deprecated, Cloudflare-blocked, or non-existent)
- Local neural network analysis provides accurate results
- Tool works fully offline with no API dependencies

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

### Why MSAF for structure detection?

- Music Structure Analysis Framework (ISMIR 2016), well-documented academic framework
- Lightweight dependencies (librosa, scipy, scikit-learn - no PyTorch for structure)
- Multiple boundary/labeling algorithms available for experimentation
- MIT license, compatible with project licensing
- CPU-only, no GPU/CUDA complexity

Previous versions used Allin1 but it was replaced due to:
- PyTorch dependency with strict version requirements
- Required madmom (unmaintained, Python 3.10+ compatibility issues)
- NATTEN dependency with complex GPU/PyTorch version matrix
- Heavy installation (~2GB+ with PyTorch)

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| CLI | typer + rich | Modern CLI with colors/tables |
| Audio Analysis | beat_this, librosa, msaf | BPM detection, structure analysis, audio loading |
| Audio I/O | mutagen | Metadata reading |
| Config | pydantic | Validation, env vars |
| Logging | structlog | Structured logging |
| Testing | pytest | Test framework |
| Linting | ruff | Fast linting |
| Type Checking | mypy | Static analysis |

## Placeholder / Unimplemented Features

The following features are documented but currently return hardcoded/placeholder values:

### Configuration File Support (`src/edm/config.py`)

**Status:** Partial implementation

- TOML configuration file path is recognized and logged
- File loading and parsing is not yet implemented
- Code currently returns default configuration regardless of file contents
- **TODO:** Complete TOML parsing and configuration loading
