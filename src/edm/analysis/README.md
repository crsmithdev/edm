# Analysis Module

Audio analysis implementations for BPM detection, structure analysis, and bar calculations.

## Overview

The analysis module provides high-level APIs for analyzing EDM tracks:

- **BPM Detection**: Tempo estimation with cascading fallback (metadata → neural network → spectral analysis)
- **Structure Analysis**: Section detection (intro, buildup, drop, breakdown, outro) with bar positions
- **Bar Calculations**: Convert time positions to musical bars based on BPM and time signature
- **Beat Detection**: Beat grid generation for precise bar alignment

## Module Organization

```
src/edm/analysis/
├── bpm.py                # Public API: analyze_bpm()
├── bpm_detector.py       # Detectors: beat_this, librosa
├── structure.py          # Public API: analyze_structure()
├── structure_detector.py # Detectors: MSAF, energy-based
├── bars.py               # Bar calculations (time ↔ bars)
├── beat_detector.py      # Beat tracking (grid generation)
├── beat_grid.py          # Beat grid alignment (future)
└── __init__.py           # Module exports
```

## Two-Tier Architecture

### Public APIs (Tier 1)

High-level functions for end users:

**`bpm.py`**:
- `analyze_bpm(filepath, **options) -> BPMResult`
- Cascading strategy: metadata → beat_this → librosa
- Simple interface, stable across detector changes

**`structure.py`**:
- `analyze_structure(filepath, detector="auto") -> StructureResult`
- Detector selection: auto, msaf, energy
- Includes section detection + bar calculation

**`bars.py`**:
- `time_to_bars(time, bpm, time_signature) -> float`
- `bars_to_time(bar, bpm, time_signature) -> float`
- `bar_count_for_range(start, end, bpm, time_signature) -> float`

### Detectors (Tier 2)

Low-level algorithm implementations:

**`bpm_detector.py`**:
- `compute_bpm_beat_this(filepath) -> ComputedBPM`
- `compute_bpm_librosa(filepath) -> ComputedBPM`
- Pure implementations (no fallback logic)

**`structure_detector.py`**:
- `MSAFDetector.detect(filepath) -> list[DetectedSection]`
- `EnergyDetector.detect(filepath) -> list[DetectedSection]`
- Protocol-based interface for extensibility

## Data Flow

### BPM Analysis

```
analyze_bpm(filepath)
    ↓
1. Try metadata (ID3/FLAC tags)
    ↓ (if missing)
2. compute_bpm_beat_this(filepath)
    ↓ (if fails/unavailable)
3. compute_bpm_librosa(filepath)
    ↓
BPMResult(bpm, confidence, source, method)
```

### Structure Analysis

```
analyze_structure(filepath)
    ↓
1. Select detector (auto → MSAF if available → energy)
    ↓
2. Detector.detect(filepath) → list[DetectedSection]
    ↓
3. Get BPM via analyze_bpm(filepath)
    ↓
4. Calculate bars for each section (time_to_bars)
    ↓
5. Convert to Section with bar positions
    ↓
StructureResult(sections, events, raw, duration, bpm)
```

### Bar Calculation

```
Section(start_time=30.0, end_time=90.0)
    ↓
BPM = 128, time_signature = (4, 4)
    ↓
start_bar = time_to_bars(30.0, 128, (4, 4)) = 16.0
end_bar = time_to_bars(90.0, 128, (4, 4)) = 48.0
bar_count = 48.0 - 16.0 = 32.0
    ↓
Section(start_bar=16.0, end_bar=48.0, bar_count=32.0)
```

## Key Types

### BPMResult

```python
@dataclass
class BPMResult:
    bpm: float                      # Detected BPM
    confidence: float               # 0-1 confidence score
    source: "metadata" | "computed" # Data source
    method: str | None              # Detection method (beat-this, librosa, id3)
    computation_time: float         # Seconds spent computing
    alternatives: list[float]       # Tempo multiplicity candidates
```

### Section

```python
@dataclass
class Section:
    label: str              # intro, buildup, drop, breakdown, outro
    start_time: float       # Seconds
    end_time: float         # Seconds
    confidence: float       # 0-1 confidence score
    start_bar: float | None # 1-indexed bar position (fractional)
    end_bar: float | None   # 1-indexed bar position
    bar_count: float | None # Number of bars in section
```

### StructureResult

```python
@dataclass
class StructureResult:
    sections: list[Section]       # Chronological span sections
    events: list[tuple[float, str]] # Moment-based events (bar, label)
    raw: list[RawSection]         # Debug detail (full detector output)
    duration: float               # Total track seconds
    detector: str                 # msaf, energy
    bpm: float | None             # BPM used for bar calculations
    downbeat: float               # First beat time (seconds)
    time_signature: tuple[int, int] # (numerator, denominator)
```

## Algorithms

### BPM Detection

See `docs/analysis-algorithms.md` for technical details.

**beat_this** (neural network):
- Pros: Highest accuracy, handles complex rhythms
- Cons: Slow initialization (~5s), GPU memory
- Use: EDM tracks, accuracy critical

**librosa** (spectral flux):
- Pros: Fast, CPU-only, well-tested
- Cons: Less accurate for EDM
- Use: Speed priority, non-EDM tracks

### Structure Detection

**MSAF** (Music Structure Analysis Framework):
- Spectral flux boundary detection
- Energy-based EDM label mapping
- Pros: Accurate boundaries, academic research
- Cons: Slower than energy detector

**Energy** (rule-based):
- RMS energy gradient analysis
- Energy percentile thresholds
- Pros: Fast, no ML dependencies
- Cons: Less accurate than MSAF

### Bar Calculations

**Formula**:
```python
bar_position = (time_seconds * bpm / 60.0) / beats_per_bar
```

**Features**:
- 1-indexed bars (bar 1 = first bar)
- Fractional positions (16.5 = halfway through bar 17)
- Graceful degradation (None if BPM unavailable)
- Time signature support (4/4, 3/4, 6/8, etc.)

## Relationships

### Dependencies

```
bpm.py → bpm_detector.py → io/audio.py
structure.py → structure_detector.py → io/audio.py
structure.py → bpm.py (for bar calculations)
structure.py → bars.py (time ↔ bar conversion)
```

### Detector Protocol

Detectors implement a common interface:

```python
class StructureDetector(Protocol):
    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        ...
```

Adding new detectors:
1. Implement `detect()` method
2. Return `list[DetectedSection]`
3. Register in `get_detector()` function

Example:

```python
class CustomDetector:
    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        # Custom algorithm
        return [DetectedSection(...), ...]

def get_detector(name: str) -> StructureDetector:
    if name == "custom":
        return CustomDetector()
    # ...existing detectors
```

## Usage Examples

### Basic BPM Detection

```python
from pathlib import Path
from edm.analysis.bpm import analyze_bpm

result = analyze_bpm(Path("track.mp3"))
print(f"BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")
print(f"Source: {result.source}, Method: {result.method}")
```

### Structure with Bars

```python
from edm.analysis.structure import analyze_structure

result = analyze_structure(Path("track.mp3"))

for section in result.sections:
    if section.start_bar and section.end_bar:
        print(f"{section.label}: bars {int(section.start_bar)}-{int(section.end_bar)}")
    else:
        print(f"{section.label}: {section.start_time:.1f}s-{section.end_time:.1f}s")
```

### Force Specific Detector

```python
# Force MSAF detector
result = analyze_structure(Path("track.mp3"), detector="msaf")

# Force energy detector (faster)
result = analyze_structure(Path("track.mp3"), detector="energy")

# Auto selection (default)
result = analyze_structure(Path("track.mp3"), detector="auto")
```

### Manual Bar Calculations

```python
from edm.analysis.bars import time_to_bars, bars_to_time

# Convert time to bars
bpm = 128.0
time_sig = (4, 4)  # 4/4 time

bar_position = time_to_bars(30.0, bpm, time_sig)
print(f"30.0s = bar {bar_position:.2f}")  # "30.0s = bar 16.00"

# Convert bars back to time
time_position = bars_to_time(16.0, bpm, time_sig)
print(f"Bar 16 = {time_position:.1f}s")  # "Bar 16 = 30.0s"
```

### Ignore Metadata

```python
# Force computation (skip metadata)
result = analyze_bpm(Path("track.mp3"), ignore_metadata=True)
```

## Error Handling

### Exceptions

All exceptions derive from `EDMError` (see `src/edm/exceptions.py`):

- `AudioFileError`: File not found, corrupt audio
- `AnalysisError`: Analysis failed (insufficient beats, etc.)
- `BPMDetectionError`: Specific to BPM detection failures
- `StructureDetectionError`: Specific to structure analysis failures

### Graceful Degradation

Functions return partial results when possible:

```python
# If BPM detection fails, bar fields are None
result = analyze_structure(Path("track.mp3"))
section = result.sections[0]

if section.start_bar is None:
    # Fall back to time-based representation
    print(f"{section.label}: {section.start_time}s-{section.end_time}s")
else:
    # Use bar-based representation
    print(f"{section.label}: bars {int(section.start_bar)}-{int(section.end_bar)}")
```

## Testing

Unit tests: `tests/unit/`

```bash
# Test BPM detection
uv run pytest tests/unit/test_bpm.py -v

# Test structure analysis
uv run pytest tests/unit/test_structure.py -v

# Test bar calculations
uv run pytest tests/unit/test_bars.py -v
```

Integration tests: `tests/integration/`

```bash
# Full pipeline tests with fixtures
uv run pytest tests/integration/ -v
```

## Performance Constraints

From `docs/architecture.md`:

- **Max processing time**: 30s per track
- **Memory limit**: 4GB
- **GPU support**: Optional (CUDA for beat_this)

Typical performance:

| Operation | Time | Memory |
|-----------|------|--------|
| BPM (metadata) | <10ms | <1MB |
| BPM (beat_this) | 5-10s | ~500MB |
| BPM (librosa) | 2-3s | ~100MB |
| Structure (MSAF) | 3-5s | ~200MB |
| Structure (energy) | 1-2s | ~100MB |
| Bar calculation | <1ms | <1KB |

## Future Enhancements

### Beat Grid Alignment

Placeholder support exists in `beat_grid.py`:

**Current**: Bars calculated from BPM only (assumes downbeat at 0.0s)

**Future**: Detect first downbeat, align bars to downbeat
- Use beat_this downbeat detection
- Adjust calculations: `bar = (time - downbeat) * bpm / 60.0 / beats_per_bar`
- Improve accuracy for tracks with intro silence

**API** (reserved parameter):
```python
analyze_structure(filepath, beat_grid=True)  # Enable beat grid alignment
```

### Additional Detectors

Easy to add new structure detectors:
1. Implement `StructureDetector` protocol
2. Return `list[DetectedSection]`
3. Register in `get_detector()`

Potential additions:
- Essentia-based detection
- Specialized drop detection
- Harmonic change detection

## See Also

- `docs/analysis-algorithms.md`: Technical algorithm details
- `docs/architecture.md`: System architecture and design patterns
- `src/edm/io/audio.py`: Audio loading and caching
- `src/edm/evaluation/`: Accuracy evaluation framework
