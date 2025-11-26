# Design: Replace madmom with beat_this

## Context

madmom has been the primary BPM detector but is unmaintained:
- Last PyPI release: 2019
- Python version constraint: <3.10
- Cython/NumPy version conflicts during installation
- No active development or bug fixes

beat_this is the successor from the same research group (CPJKU):
- ISMIR 2024 paper: "Accurate Beat Tracking Without DBN Postprocessing"
- MIT licensed, actively maintained
- Python 3.10+ support
- State-of-the-art accuracy

## Goals / Non-Goals

**Goals:**
- Replace madmom with beat_this for primary BPM detection
- Maintain or improve BPM detection accuracy
- Support modern Python versions (3.10+)
- Keep librosa as lightweight fallback

**Non-Goals:**
- Real-time/streaming beat detection (future scope)
- Downbeat detection UI (beat_this provides downbeats but we don't expose them yet)
- Supporting madmom as an alternative

## Decisions

### Decision 1: Use beat_this via GitHub install

beat_this is not on PyPI. Install from GitHub:
```
pip install https://github.com/CPJKU/beat_this/archive/main.zip
```

**Alternatives considered:**
- Wait for PyPI release: Unknown timeline, blocks progress
- Vendor the code: Maintenance burden, license complexity
- Fork and publish: Unnecessary overhead

### Decision 2: Calculate BPM from beat positions

beat_this returns beat timestamps, not BPM directly. Calculate BPM from inter-beat intervals:

```python
beats, downbeats = file2beats(filepath)
intervals = np.diff(beats)
bpm = 60.0 / np.median(intervals)
```

This matches our existing madmom approach and allows confidence calculation from interval consistency.

### Decision 3: Device selection with graceful fallback

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

CPU inference is slower but functional. No hard CUDA requirement.

### Decision 4: Model variant selection

Use `final0` checkpoint (78MB) by default. Consider `small` (8.1MB) as optional for resource-constrained environments.

### Decision 5: Maintain method signature compatibility

Keep `prefer_madmom` parameter but rename semantics:
- `prefer_madmom=True` → Use beat_this (neural network approach)
- `prefer_madmom=False` → Use librosa (traditional DSP)

Consider deprecating the parameter name in a future change.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| beat_this not on PyPI | Pin to specific commit hash in pyproject.toml |
| PyTorch adds ~2GB to install | Already have PyTorch for other ML features |
| Model download on first run | beat_this bundles models, no download needed |
| GPU memory for large files | CPU fallback works, just slower |

## Migration Plan

1. Add beat_this dependency
2. Implement `compute_bpm_beat_this()` parallel to madmom
3. Switch default in `compute_bpm()`
4. Remove madmom code and dependency
5. Update documentation

**Rollback:** Revert to previous commit. madmom code preserved in git history.

## Open Questions

- Should we expose downbeat information from beat_this? (Deferred to future change)
- Should `prefer_madmom` parameter be renamed to `prefer_neural`? (Deferred)
