# Development Guide

## Setup

### Prerequisites

- Python 3.12+
- ffmpeg (required for audio loading)
- System packages (Ubuntu/Debian): `python3-dev`, `build-essential`

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify
uv run edm --version
```

## Testing

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_analysis/test_bpm.py

# Run specific test
uv run pytest tests/test_analysis/test_bpm.py::test_analyze_bpm_from_metadata

# Coverage report
uv run pytest --cov=src --cov-report=term-missing
```

### Test Structure

Tests mirror `src/` structure:

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data
│   └── reference/           # Reference data files
├── test_analysis/           # Tests for src/edm/analysis/
│   ├── test_bpm.py
│   └── test_bpm_detector.py
├── test_evaluation/         # Tests for src/edm/evaluation/
└── ...
```

### Test Fixtures

Audio files for testing: `~/music`

Fixtures defined in `tests/conftest.py`:
- `tmp_audio_file` - Temporary audio file
- `sample_metadata` - Sample track metadata

### Writing Tests

```python
import pytest
from pathlib import Path
from edm.analysis.bpm import analyze_bpm

def test_analyze_bpm_returns_result():
    """Test that analyze_bpm returns a BPMResult."""
    result = analyze_bpm(Path("tests/fixtures/track.mp3"))
    assert result.bpm > 0
    assert 0 <= result.confidence <= 1

def test_analyze_bpm_raises_on_invalid_file():
    """Test that analyze_bpm raises AnalysisError for invalid files."""
    with pytest.raises(AnalysisError, match="cannot be loaded"):
        analyze_bpm(Path("nonexistent.mp3"))

class TestBPMStrategy:
    """Tests for BPM lookup strategy."""

    def test_metadata_first(self, track_with_metadata):
        result = analyze_bpm(track_with_metadata)
        assert result.source == "metadata"

    def test_offline_skips_spotify(self, track_without_metadata):
        result = analyze_bpm(track_without_metadata, offline=True)
        assert result.source in ["metadata", "computed"]
```

## Code Quality

### Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

Configuration in `pyproject.toml`:
- Line length: 100
- Target: Python 3.12+
- Rules: E, F, I, N, W

### Type Checking

```bash
# Run mypy
uv run mypy src/

# Ignore missing imports (for untyped libraries)
uv run mypy src/ --ignore-missing-imports
```

### Dead Code Detection

```bash
vulture src/ tests/ --min-confidence 60
```

## Logging

### Configuration

Uses structlog (`src/edm/logging.py:27`):

```python
from edm.logging import configure_logging, get_logger

# Configure at startup
configure_logging(
    level="DEBUG",           # DEBUG, INFO, WARNING, ERROR
    json_format=False,       # True for JSON output
    log_file=Path("app.log"), # Optional file output
    no_color=False           # Disable colors
)

# Get a logger
logger = get_logger(__name__)
```

### Usage Patterns

```python
import structlog

logger = structlog.get_logger(__name__)

# Basic logging
logger.info("processing started", file_path="track.mp3")
logger.debug("attempting metadata lookup", filepath=str(path))
logger.warning("metadata lookup failed", error=str(e))
logger.error("analysis failed", filepath=str(path), error=str(e))

# With context
logger.info(
    "bpm_computed",
    filepath=str(filepath),
    bpm=round(result.bpm, 1),
    method=result.method,
    confidence=round(result.confidence, 2),
)
```

### CLI Logging Options

```bash
# Set log level
uv run edm analyze track.mp3 --log-level DEBUG

# JSON format (for log aggregation)
uv run edm analyze track.mp3 --json-logs

# Write to file
uv run edm analyze track.mp3 --log-file analysis.log

# Disable colors
uv run edm analyze track.mp3 --no-color
```

## Evaluation Framework

### Running Evaluations

```bash
# Against file metadata
uv run edm evaluate bpm --source ~/music --reference metadata

# Against CSV reference
uv run edm evaluate bpm --source ~/music --reference data/annotations/bpm_tagged.csv

# Full evaluation with seed
uv run edm evaluate bpm --source ~/music --reference metadata --full --seed 42
```

### Reference Sources

| Source | Description | File |
|--------|-------------|------|
| `metadata` | File ID3 tags | - |
| CSV | File with `file,bpm` columns | `src/edm/evaluation/reference.py:63` |
| JSON | File-to-BPM mapping | `src/edm/evaluation/reference.py:105` |

### Metrics

Calculated in `src/edm/evaluation/common.py`:

| Metric | Function | Description |
|--------|----------|-------------|
| MAE | `calculate_mae()` | Mean Absolute Error |
| RMSE | `calculate_rmse()` | Root Mean Square Error |
| Accuracy | `calculate_accuracy_within_tolerance()` | % within tolerance |

### Results

Output to `data/accuracy/bpm/`:
- `<timestamp>_<commit>.json` - Full results
- `<timestamp>_<commit>.md` - Summary
- `latest.*` - Symlinks to most recent

## Development Workflow

### Before Committing

```bash
# Run all checks
uv run pytest                          # Tests pass
uv run mypy src/                       # Types check
uv run ruff check .                    # Linting passes
```

### Adding a Feature

1. Create OpenSpec proposal (if significant)
2. Write tests first (TDD encouraged)
3. Implement feature
4. Update documentation if needed
5. Run all checks
6. Commit with descriptive message

### Debugging

```bash
# Verbose output
uv run edm analyze track.mp3 --log-level DEBUG --no-color

# Log to file for analysis
uv run edm analyze track.mp3 --log-file debug.log --json-logs
```

## Quick Reference

```bash
uv sync                                      # Install dependencies
uv run edm analyze track.mp3                 # Analyze file
uv run edm evaluate bpm --source ~/music --reference metadata  # Evaluate
uv run ruff check --fix . && ruff format .   # Lint and format
uv run mypy src/                             # Type check
uv run pytest -v                             # Run tests
uv run pytest --cov=src --cov-report=term    # Tests with coverage
```
