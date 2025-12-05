# Testing Guide

## Framework

- pytest with fixtures
- Name test files `test_*.py`, test functions `test_*`
- Group related tests in classes prefixed with `Test`
- Use `pytest.raises` for exception testing with `match=` for message validation

## Running Tests

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_bpm_detector.py

# Run specific test class
uv run pytest tests/unit/test_bars.py::TestTimeToBar

# Run specific test
uv run pytest tests/unit/test_bpm_detector.py::test_compute_bpm_returns_result

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test audio files
│   ├── generate_test_audio.py  # Audio generator script
│   ├── tag_test_audio.py       # Metadata tagging script
│   ├── 120bpm_4beat.wav        # 120 BPM test audio
│   ├── 128bpm_4beat.wav        # 128 BPM test audio
│   └── ...                     # Various BPM/pattern fixtures
├── estimations/             # Reference JAMS annotations
│   └── *.jams               # Beat pattern annotations
├── unit/                    # Unit tests (~270 tests)
│   ├── test_analysis.py     # BPM analysis tests
│   ├── test_audio_cache.py  # Audio caching tests
│   ├── test_bars.py         # Bar calculation tests
│   ├── test_beat_detector.py # Beat detection tests
│   ├── test_beat_grid.py    # Beat grid tests
│   ├── test_bpm_detector.py # BPM detector tests
│   ├── test_config.py       # Configuration tests
│   ├── test_logging.py      # Logging configuration tests
│   ├── test_metadata.py     # Metadata extraction tests
│   ├── test_output_formats.py # Output format tests
│   ├── test_structure.py    # Structure analysis tests
│   ├── processing/          # Processing module tests
│   │   └── test_parallel.py # Parallel processing tests
│   └── test_evaluation/     # Evaluation framework tests
│       ├── test_bpm.py      # BPM evaluation tests
│       ├── test_common.py   # Common utilities tests
│       ├── test_reference.py # Reference data tests
│       └── test_structure.py # Structure evaluation tests
├── integration/             # Integration tests (~3 tests)
│   ├── test_cli.py          # CLI integration tests
│   └── test_analyze_parallel.py # Parallel analysis tests
└── performance/             # Performance benchmarks (~6 tests)
    └── test_parallel_speedup.py # Speedup measurement
```

## Test Fixtures

### Audio Files

Test audio files in `tests/fixtures/`:

| File | BPM | Pattern | Purpose |
|------|-----|---------|---------|
| `120bpm_4beat.wav` | 120 | 4/4 | Standard tempo test |
| `128bpm_4beat.wav` | 128 | 4/4 | EDM standard tempo |
| `140bpm_4beat.wav` | 140 | 4/4 | High tempo test |
| Various FLAC files | Mixed | Mixed | Format testing |

Generate new fixtures:
```bash
python tests/fixtures/generate_test_audio.py
```

### Reference Annotations

JAMS files in `tests/estimations/` for beat evaluation:
- Contains beat timestamps and tempo annotations
- Used by beat detector accuracy tests

### Common Patterns

```python
from pathlib import Path

# Define fixture path at module level
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
ESTIMATIONS_DIR = Path(__file__).parent.parent / "estimations"

def test_with_fixture():
    audio_file = FIXTURES_DIR / "128bpm_4beat.wav"
    # ... test code
```

## Writing Tests

### Basic Test

```python
import pytest
from pathlib import Path
from edm.analysis.bpm import analyze_bpm

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def test_analyze_bpm_returns_result():
    """Test that analyze_bpm returns a BPMResult."""
    result = analyze_bpm(FIXTURES_DIR / "128bpm_4beat.wav")
    assert result.bpm > 0
    assert 0 <= result.confidence <= 1
```

### Exception Testing

```python
import pytest
from edm.exceptions import AnalysisError

def test_raises_on_invalid_file():
    """Test that invalid files raise AnalysisError."""
    with pytest.raises(AnalysisError, match="cannot be loaded"):
        analyze_bpm(Path("nonexistent.mp3"))
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("bpm,expected", [
    (120, 120.0),
    (128, 128.0),
    (140, 140.0),
])
def test_bpm_detection_accuracy(bpm, expected):
    """Test BPM detection across various tempos."""
    result = analyze_bpm(FIXTURES_DIR / f"{bpm}bpm_4beat.wav")
    assert abs(result.bpm - expected) <= 2.0  # Within 2 BPM tolerance
```

### Test Classes

```python
class TestBPMStrategy:
    """Tests for BPM lookup strategy."""

    def test_metadata_first(self):
        """Metadata source is preferred when available."""
        result = analyze_bpm(FIXTURES_DIR / "tagged_128bpm.flac")
        assert result.source == "metadata"

    def test_computed_fallback(self):
        """Falls back to computation when no metadata."""
        result = analyze_bpm(FIXTURES_DIR / "untagged.wav")
        assert result.source == "computed"
```

### Markers

```python
import pytest

@pytest.mark.slow
def test_large_file_processing():
    """Test processing of large audio files."""
    # ... slow test

@pytest.mark.integration
def test_full_pipeline():
    """Integration test for full analysis pipeline."""
    # ... integration test
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions and classes in isolation:
- Mock external dependencies
- Fast execution (<1s per test)
- High coverage of edge cases

### Integration Tests (`tests/integration/`)

Test component interactions:
- Real file I/O
- CLI commands
- Parallel processing

### Performance Tests (`tests/performance/`)

Benchmark performance characteristics:
- Speedup measurements
- Memory usage
- Processing time

Run with:
```bash
uv run pytest tests/performance/ -v --benchmark-only
```

## Coverage

Check coverage:
```bash
# Terminal report
uv run pytest --cov=src --cov-report=term-missing

# HTML report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

Target: 80%+ coverage on core modules (`analysis/`, `evaluation/`)

## Continuous Integration

Tests run on every commit via GitHub Actions:
- All unit tests
- Integration tests
- Linting and type checking

See `.github/workflows/` for CI configuration.
