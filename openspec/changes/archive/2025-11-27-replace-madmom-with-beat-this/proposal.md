# Change: Replace madmom with beat_this for BPM Detection

## Why

madmom is unmaintained (last PyPI release 2019, Python <3.10 only) and causes persistent installation failures due to Cython/NumPy version conflicts. beat_this is actively maintained (ISMIR 2024), achieves state-of-the-art accuracy without DBN postprocessing, and supports modern Python versions.

## What Changes

- **BREAKING**: Remove madmom dependency entirely
- Replace `compute_bpm_madmom()` with `compute_bpm_beat_this()` in `bpm_detector.py`
- Add beat_this as new dependency (PyTorch-based)
- Update `project.md` tech stack documentation to reflect beat_this as primary BPM detector
- Maintain librosa as fallback for environments without PyTorch/CUDA

## Impact

- Affected specs: `analysis`, `core-library`
- Affected code:
  - `src/edm/analysis/bpm_detector.py` - Primary implementation
  - `src/edm/analysis/bpm.py` - Method naming in docstrings
  - `pyproject.toml` - Dependency swap
  - `openspec/project.md` - Tech stack update
  - `tests/test_analysis/test_bpm.py` - Update test expectations
