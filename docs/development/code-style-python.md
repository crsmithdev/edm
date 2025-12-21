# Python Style Guide

This guide covers code style, best practices, and patterns for Python code in the EDM project. We use Python 3.12+ and enforce strict type checking with mypy.

## Code Style

### Formatter and Linter

Use **Ruff** for both formatting and linting:

```bash
uv run ruff format .        # Format code
uv run ruff check --fix .   # Lint with auto-fix
```

Run both together:

```bash
uv run ruff check --fix . && ruff format .
```

### Quality Checks

Run all quality checks before committing:

```bash
# Formatting
uv run ruff format .

# Linting
uv run ruff check --fix .

# Type checking
uv run mypy packages/edm-lib/src/

# Tests
uv run pytest -v

# All together (recommended)
just check
```

### Line Length

100 characters (configured in pyproject.toml)

### Quotes

Double quotes preferred for strings:

**Good:**
```python
name = "EDM"
message = "BPM detection complete"
```

**Bad:**
```python
name = 'EDM'
message = 'BPM detection complete'
```

### Import Organization

stdlib → third-party → local, sorted alphabetically within groups:

**Good:**
```python
import os
from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel

from edm.analysis import bpm
from edm.config import settings
```

**Bad:**
```python
from edm.config import settings
import numpy as np
import os
from edm.analysis import bpm
from pydantic import BaseModel
import librosa
from pathlib import Path
```

## Type Hints

Always use type hints for function signatures. Use modern Python 3.12+ syntax throughout:

**Good:**
```python
from collections.abc import Sequence
from typing import TypeAlias

# Modern syntax
def process_files(paths: list[str], count: int | None = None) -> dict[str, int]:
    """Process files and return counts."""
    results = {}
    for path in paths:
        results[path] = count or 0
    return results

# Complex types with TypeAlias
AudioData: TypeAlias = tuple[np.ndarray, int]

def load_audio(path: str) -> AudioData:
    """Load audio file and return samples with sample rate."""
    y, sr = librosa.load(path)
    return y, sr
```

**Bad:**
```python
from typing import List, Optional, Dict, Tuple

# Deprecated syntax
def process_files(paths: List[str], count: Optional[int] = None) -> Dict[str, int]:
    results = {}
    for path in paths:
        results[path] = count or 0
    return results

# No TypeAlias for complex types
def load_audio(path: str) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path)
    return y, sr
```

### Type Checking

Run mypy in strict mode:

```bash
uv run mypy --strict src/
```

## Error Handling

Create a base exception and derive specific exceptions from it:

**Good:**
```python
class EDMError(Exception):
    """Base exception for EDM."""
    pass

class AudioLoadError(EDMError):
    """Failed to load audio file."""
    pass

class BPMDetectionError(EDMError):
    """BPM detection failed."""
    pass

# Usage
try:
    y, sr = librosa.load(path)
except Exception as e:
    raise AudioLoadError(f"Failed to load {path}") from e
```

**Bad:**
```python
# No base exception
class AudioLoadError(Exception):
    pass

# Bare except
try:
    y, sr = librosa.load(path)
except:
    raise AudioLoadError(f"Failed to load {path}")

# Not preserving exception chain
try:
    y, sr = librosa.load(path)
except Exception as e:
    raise AudioLoadError(f"Failed to load {path}")
```

Always be specific in except clauses and use `raise ... from e` to preserve exception chains.

## Async Patterns

### HTTP Requests

Prefer `httpx` over `requests` for HTTP (supports async):

**Good:**
```python
import httpx

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```

**Bad:**
```python
import requests

def fetch_data(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Concurrent Tasks

Use `asyncio.TaskGroup` (Python 3.12+) for concurrent tasks:

**Good:**
```python
import asyncio

async def process_multiple(paths: list[str]) -> list[float]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(analyze_file(path)) for path in paths]
    return [task.result() for task in tasks]
```

**Bad:**
```python
import asyncio

async def process_multiple(paths: list[str]) -> list[float]:
    # asyncio.gather is less safe (doesn't cancel on failure)
    return await asyncio.gather(*[analyze_file(path) for path in paths])
```

Avoid mixing sync and async code paths - choose one approach per module.

## Data Validation

Use Pydantic v2 for data models and validation:

**Good:**
```python
from pydantic import BaseModel, Field, model_validator

class BPMResult(BaseModel):
    """BPM detection result."""
    bpm: float = Field(gt=0, le=300, description="Detected BPM")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    file_path: str

    @model_validator(mode="after")
    def validate_confidence(self) -> "BPMResult":
        if self.confidence < 0.5 and self.bpm > 200:
            raise ValueError("High BPM requires high confidence")
        return self
```

**Bad:**
```python
from pydantic import BaseModel, validator

class BPMResult(BaseModel):
    bpm: float
    confidence: float
    file_path: str

    @validator("confidence")  # Deprecated in v2
    def validate_confidence(cls, v, values):
        if v < 0.5 and values.get("bpm", 0) > 200:
            raise ValueError("High BPM requires high confidence")
        return v
```

Key points:
- Prefer `model_validator` over deprecated `root_validator`/`validator`
- Use `Field()` for constraints and documentation
- Pydantic v2 has better performance and type checking

## Logging

Use `logging` stdlib, configure once at entry point:

**Good:**
```python
import logging

logger = logging.getLogger(__name__)

def analyze_bpm(path: str) -> float:
    logger.info(f"Starting BPM analysis for {path}")
    try:
        result = _detect_bpm(path)
        logger.info(f"BPM detection complete: {result:.1f} BPM")
        return result
    except Exception as e:
        logger.error(f"BPM detection failed: {e}")
        raise
```

**Bad:**
```python
import logging

def analyze_bpm(path: str) -> float:
    # No module-level logger
    logging.info("start_bpm_analysis")  # Not natural language
    try:
        result = _detect_bpm(path)
        logging.info(f"bpm_detection_complete {result}")  # Inconsistent format
        return result
    except Exception as e:
        logging.error("error")  # Not descriptive
        raise
```

Key points:
- Use `logger = logging.getLogger(__name__)` per module
- Log in natural language: "Starting BPM analysis" not "start_bpm_analysis"
- Use appropriate capitalization and punctuation
- Prefer f-strings in log calls only when level is enabled (or use lazy `%` formatting)

## Naming Conventions

### General Rules

Follow PEP 8 naming conventions:

**Good:**
```python
# Variables and functions: snake_case
audio_file = "track.mp3"
def analyze_bpm(path: str) -> float: ...

# Classes: PascalCase
class BPMDetector: ...
class AudioLoadError(Exception): ...

# Constants: UPPER_SNAKE_CASE
MAX_BPM = 200
DEFAULT_SAMPLE_RATE = 22050

# Private: leading underscore
def _internal_helper() -> None: ...
_cache: dict[str, Any] = {}

# Type aliases: PascalCase
AudioData: TypeAlias = tuple[np.ndarray, int]
```

**Bad:**
```python
# Wrong conventions
AudioFile = "track.mp3"  # Should be audio_file
def AnalyzeBPM(path: str) -> float: ...  # Should be analyze_bpm
class bpm_detector: ...  # Should be BPMDetector
maxBPM = 200  # Should be MAX_BPM
```

### Descriptive Names

Use clear, descriptive names:

**Good:**
```python
def calculate_tempo_from_onset_strength(
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_length: int
) -> float:
    """Calculate tempo from onset strength envelope."""
    autocorrelation = librosa.autocorrelate(onset_envelope)
    tempo_candidates = _extract_tempo_candidates(autocorrelation)
    return _select_most_likely_tempo(tempo_candidates)
```

**Bad:**
```python
def calc(x: np.ndarray, sr: int, hl: int) -> float:
    ac = librosa.autocorrelate(x)
    tc = _extract(ac)
    return _select(tc)
```

## Code Organization

### Module Structure

Organize modules consistently:

```python
"""Module docstring describing purpose."""

# Standard library imports
import os
from pathlib import Path
from typing import TypeAlias

# Third-party imports
import librosa
import numpy as np
from pydantic import BaseModel

# Local imports
from edm.analysis import BPMDetector
from edm.config import settings
from edm.exceptions import AudioLoadError

# Type aliases
AudioData: TypeAlias = tuple[np.ndarray, int]

# Constants
DEFAULT_SAMPLE_RATE = 22050
MAX_BPM = 200

# Public API
__all__ = ["analyze_audio", "BPMResult"]

# Module-level logger
logger = logging.getLogger(__name__)

# Classes and functions
class BPMResult(BaseModel):
    ...

def analyze_audio(path: str) -> BPMResult:
    ...

# Private helpers
def _load_audio_data(path: str) -> AudioData:
    ...
```

### File Organization

Keep files focused and manageable:

**Good:**
```
src/edm/analysis/
├── __init__.py          # Public API exports
├── bpm.py               # BPM detection (<300 lines)
├── structure.py         # Structure analysis (<300 lines)
├── beat_grid.py         # Beat grid generation (<200 lines)
└── _helpers.py          # Internal utilities
```

**Bad:**
```
src/edm/
└── analysis.py          # 2000+ lines, everything mixed
```

### Class Design

Keep classes focused with single responsibility:

**Good:**
```python
class BPMDetector:
    """Detects BPM from audio using onset strength."""

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def detect(self, audio: np.ndarray) -> float:
        """Detect BPM from audio signal."""
        onset_env = self._compute_onset_strength(audio)
        tempo = self._estimate_tempo(onset_env)
        return self._refine_tempo(tempo, onset_env)

    def _compute_onset_strength(self, audio: np.ndarray) -> np.ndarray:
        """Compute onset strength envelope."""
        return librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

    def _estimate_tempo(self, onset_env: np.ndarray) -> float:
        """Estimate tempo from onset envelope."""
        return float(librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0])

    def _refine_tempo(self, tempo: float, onset_env: np.ndarray) -> float:
        """Refine tempo estimate using autocorrelation."""
        # Refinement logic
        return tempo
```

**Bad:**
```python
class AudioAnalyzer:
    """Does everything related to audio."""  # Too broad

    def analyze(self, path: str):  # No type hints
        # 500+ lines of mixed logic
        # BPM, structure, beats all in one method
        pass
```

## Performance

### NumPy Optimization

Use vectorized operations:

**Good:**
```python
# Vectorized
normalized = (audio - audio.mean()) / audio.std()
energies = np.sum(spectrogram ** 2, axis=0)

# Avoid loops for large arrays
differences = np.diff(onset_times)
valid_onsets = onset_times[differences > min_interval]
```

**Bad:**
```python
# Slow loop
normalized = np.array([
    (x - audio.mean()) / audio.std()
    for x in audio
])

# Manual iteration
energies = []
for col in range(spectrogram.shape[1]):
    energies.append(sum(spectrogram[:, col] ** 2))
```

### Caching

Use `functools.cache` or `lru_cache` for expensive computations:

**Good:**
```python
from functools import cache, lru_cache

@cache
def load_model(model_path: str) -> torch.nn.Module:
    """Load model (cached after first call)."""
    return torch.load(model_path)

@lru_cache(maxsize=128)
def compute_spectrogram(audio_hash: str, sr: int) -> np.ndarray:
    """Compute spectrogram (cached for recent calls)."""
    audio = _load_from_hash(audio_hash)
    return librosa.stft(audio, sr=sr)
```

**Bad:**
```python
# No caching, reloads every time
def load_model(model_path: str) -> torch.nn.Module:
    return torch.load(model_path)
```

### Lazy Evaluation

Avoid unnecessary computation:

**Good:**
```python
def analyze_files(
    paths: list[str],
    max_files: int | None = None
) -> Iterator[BPMResult]:
    """Yield results lazily."""
    for i, path in enumerate(paths):
        if max_files and i >= max_files:
            break
        yield analyze_bpm(path)

# Caller can decide when to stop
for result in analyze_files(paths, max_files=10):
    print(result.bpm)
```

**Bad:**
```python
def analyze_files(paths: list[str]) -> list[BPMResult]:
    """Process all files upfront."""
    return [analyze_bpm(p) for p in paths]  # Processes everything

# No way to stop early
results = analyze_files(paths)  # Processes all 1000 files
print(results[0].bpm)  # Only needed first result
```

## Security

### Path Traversal

Validate file paths:

**Good:**
```python
from pathlib import Path

def safe_load_file(base_dir: Path, filename: str) -> str:
    """Safely load file preventing path traversal."""
    base = base_dir.resolve()
    target = (base / filename).resolve()

    # Ensure target is within base_dir
    if not target.is_relative_to(base):
        raise ValueError(f"Path traversal attempt: {filename}")

    return target.read_text()
```

**Bad:**
```python
def load_file(base_dir: str, filename: str) -> str:
    """Unsafe: allows path traversal."""
    path = os.path.join(base_dir, filename)
    return open(path).read()  # Vulnerable to ../../../etc/passwd
```

### Command Injection

Use subprocess safely:

**Good:**
```python
import subprocess
from pathlib import Path

def convert_audio(input_path: Path, output_path: Path) -> None:
    """Convert audio using ffmpeg safely."""
    subprocess.run(
        [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", "22050",
            str(output_path)
        ],
        check=True,
        capture_output=True
    )
```

**Bad:**
```python
def convert_audio(input_path: str, output_path: str) -> None:
    """Vulnerable to command injection."""
    cmd = f"ffmpeg -i {input_path} -ar 22050 {output_path}"
    os.system(cmd)  # Dangerous if paths contain shell metacharacters
```

### XML Parsing

Use defusedxml for untrusted XML:

**Good:**
```python
from defusedxml import ElementTree as DefusedET

def parse_annotation(path: str) -> dict:
    """Parse XML annotation safely."""
    tree = DefusedET.parse(path)
    root = tree.getroot()
    return _extract_data(root)
```

**Bad:**
```python
import xml.etree.ElementTree as ET

def parse_annotation(path: str) -> dict:
    """Vulnerable to XXE attacks."""
    tree = ET.parse(path)  # Dangerous with untrusted XML
    root = tree.getroot()
    return _extract_data(root)
```

### Secrets Management

Never commit secrets:

**Good:**
```python
import os
from pathlib import Path

def get_api_key() -> str:
    """Get API key from environment or secure file."""
    # Environment variable (preferred)
    if key := os.getenv("EDM_API_KEY"):
        return key

    # Secure file outside repo
    key_file = Path.home() / ".edm" / "api_key"
    if key_file.exists():
        return key_file.read_text().strip()

    raise ValueError("API key not configured")
```

**Bad:**
```python
# Hardcoded secrets (NEVER do this)
API_KEY = "sk-1234567890abcdef"  # Committed to git

def get_api_key() -> str:
    return API_KEY
```

## Testing Best Practices

### Test Organization

Organize tests to mirror source structure:

```
src/edm/analysis/
├── bpm.py
└── structure.py

tests/unit/
├── test_bpm.py
└── test_structure.py
```

### Fixture Usage

Use fixtures for common setup:

**Good:**
```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_audio_path(tmp_path: Path) -> Path:
    """Create temporary audio file for testing."""
    path = tmp_path / "test.wav"
    # Create test audio
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    sf.write(path, y, 22050)
    return path

def test_bpm_detection(sample_audio_path: Path):
    """Test BPM detection with fixture."""
    result = analyze_bpm(str(sample_audio_path))
    assert result.bpm > 0
```

**Bad:**
```python
def test_bpm_detection():
    """Inline setup in every test."""
    # Repeated in every test
    path = "/tmp/test.wav"
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    sf.write(path, y, 22050)
    result = analyze_bpm(path)
    assert result.bpm > 0
```

### Parametrized Tests

Use parametrization for multiple cases:

**Good:**
```python
@pytest.mark.parametrize("bpm,expected", [
    (60, 60.0),
    (120, 120.0),
    (180, 180.0),
])
def test_bpm_detection_accuracy(bpm: int, expected: float):
    """Test BPM detection for various tempos."""
    audio = generate_click_track(bpm)
    result = detect_bpm(audio)
    assert abs(result - expected) < 1.0
```

**Bad:**
```python
def test_bpm_60():
    audio = generate_click_track(60)
    assert abs(detect_bpm(audio) - 60.0) < 1.0

def test_bpm_120():
    audio = generate_click_track(120)
    assert abs(detect_bpm(audio) - 120.0) < 1.0

def test_bpm_180():
    audio = generate_click_track(180)
    assert abs(detect_bpm(audio) - 180.0) < 1.0
```

## Documentation

Use Google-style docstrings on public functions and classes:

**Good:**
```python
def detect_bpm(audio_path: str, method: str = "librosa") -> float:
    """Detect BPM of audio file.

    Args:
        audio_path: Path to audio file
        method: Detection method to use

    Returns:
        Detected BPM value

    Raises:
        AudioLoadError: If audio file cannot be loaded
        BPMDetectionError: If BPM detection fails
    """
    pass
```

**Bad:**
```python
def detect_bpm(audio_path: str, method: str = "librosa") -> float:
    """
    Detect BPM of audio file

    :param audio_path: str - Path to audio file
    :param method: str - Detection method to use
    :return: float - Detected BPM value
    :raises: AudioLoadError if audio file cannot be loaded
    :raises: BPMDetectionError if BPM detection fails
    """
    pass
```

Key points:
- Google style, not Sphinx/reST
- Keep docstrings to one line when possible
- Type hints replace type info in docstrings (no need to repeat types)
- Only document public APIs
