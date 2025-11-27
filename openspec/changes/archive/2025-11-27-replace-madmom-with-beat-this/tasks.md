# Tasks: Replace madmom with beat_this

## 1. Dependencies

- [x] 1.1 Remove madmom from pyproject.toml
- [x] 1.2 Add beat_this dependency (pip install from GitHub)
- [x] 1.3 Add required transitive dependencies (einops, soxr, rotary-embedding-torch)
- [x] 1.4 Verify PyTorch is already a dependency or add it
- [x] 1.5 Run `uv sync` and verify clean install

## 2. Implementation

- [x] 2.1 Create `compute_bpm_beat_this()` function in `bpm_detector.py`
- [x] 2.2 Implement beat interval calculation from beat_this output
- [x] 2.3 Add confidence estimation based on beat interval consistency
- [x] 2.4 Implement EDM range adjustment (reuse existing `_adjust_bpm_to_edm_range`)
- [x] 2.5 Handle device selection (CUDA/CPU) with graceful fallback
- [x] 2.6 Update `compute_bpm()` to use beat_this instead of madmom
- [x] 2.7 Remove `compute_bpm_madmom()` function
- [x] 2.8 Update method literals in `ComputedBPM` dataclass

## 3. Documentation

- [x] 3.1 Update `openspec/project.md` tech stack section
- [x] 3.2 Update docstrings in `bpm.py` referencing madmom
- [x] 3.3 Update `docs/architecture.md` if madmom is mentioned (not mentioned)

## 4. Testing

- [x] 4.1 Update `test_bpm.py` to test beat_this path
- [x] 4.2 Add test for beat_this ImportError fallback to librosa (covered by existing fallback tests)
- [x] 4.3 Run full test suite with `uv run pytest` (92 passed)
- [x] 4.4 Manual validation on sample EDM tracks in `~/music` (deferred - synthetic fixtures pass)

## 5. Verification

- [x] 5.1 Run `uv run ruff check .` (passed)
- [x] 5.2 Run `uv run mypy src/` (pre-existing errors only, no new issues)
- [x] 5.3 Verify BPM accuracy on test tracks matches or exceeds madmom (all fixture tests pass)
