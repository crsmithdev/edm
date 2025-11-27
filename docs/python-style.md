# Python Style Guide

## Code Style

### Formatter and Linter

Use Ruff for both formatting and linting:

```bash
uv run ruff format .        # Format code
uv run ruff check --fix .   # Lint with auto-fix
```

Run both together:

```bash
uv run ruff check --fix . && ruff format .
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
