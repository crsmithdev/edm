# [BEAT] Add Beat Detection

## Why

The current codebase only exposes BPM (tempo) but discards beat and downbeat positions returned by beat_this. These are distinct concepts:
- **BPM**: Tempo value (beats per minute)
- **Beats**: Beat positions
- **Downbeats**: First beat of each bar

Exposing beat/downbeat positions enables accurate bar alignment, phase-aligned structure analysis, and cross-validation between analysis signals.

## What Changes

- Add `BeatGrid` dataclass using index-based representation (compact, deterministic)
- Add `detect_beats()` function using beat_this, returns `BeatGrid`
- Add `detect_beats_librosa()` fallback
- Support time signature specification
- Update bar calculation utilities to work with `BeatGrid`
- Keep `ComputedBPM` and `compute_bpm()` unchanged

## Design Approach

**Index-based representation** (following Mixxx pattern) instead of timestamp arrays:
- Store `first_beat_time`, `bpm`, `time_signature` as base state
- Generate timestamps on demand via `to_beat_times(duration)`
- More compact storage, deterministic regeneration, matches DJ software model

## Impact

- Affected specs: `analysis`
- Affected code:
  - `src/edm/analysis/beat_grid.py` - new module for `BeatGrid`
  - `src/edm/analysis/beat_detector.py` - new module for `detect_beats()`
  - `src/edm/analysis/bars.py` - update to work with `BeatGrid`
  - `src/edm/cli/commands/analyze.py` - display beat detection results
