# [STRUCT]Bar Indexing and Event Format

**Status:** deployed
**Created:** 2025-12-02
**Updated:** 2025-12-03

## Why

Three issues with current structure output:

1. **Bar indexing inconsistency**: Current format uses 0-indexed bars in some contexts but DJ software (Rekordbox, Traktor, Ableton) uses 1-indexed. An 8-bar intro should be `[1, 8, intro]` not `[0, 7, intro]`, with next section starting at bar 9.

2. **Events mixed with spans**: Drops and other moment-based events (percussion onsets, bass returns) are currently output as sections with identical start/end bars, which is confusing and semantically incorrect.

3. **Redundant 'other' sections**: When multiple consecutive sections are labeled 'other', they should be merged into a single span for clarity.

## What Changes

### 1. Bar Indexing
- **BREAKING**: All bar numbers in output become 1-indexed (matching DJ software conventions)
- An 8-bar intro starting at the track beginning is `[1, 8, intro]`
- The next section starts at bar 9: `[9, 16, buildup]`
- Bar calculations internally still work from time=0, but output adds 1 to all bar indices

### 2. Event Format
- **BREAKING**: Structure output splits into two keys: `structure` (spans) and `events` (moments)
- Events use 2-element tuples: `[bar, label]`
- Spans use 3-element tuples: `[start_bar, end_bar, label]`
- Initial event labels: `drop`, `kick` (percussion onset)
- Span labels remain: `intro`, `outro`, `breakdown`, `buildup`, `verse`, `chorus`, `bridge`, `other`

### 3. Merge Consecutive 'other' Sections
- When structure detector outputs multiple consecutive sections labeled 'other', merge them into a single span
- Preserves start of first and end of last
- Uses max confidence of merged sections

## Impact

- Affected specs: `analysis`, `cli`
- Affected code:
  - `src/edm/analysis/structure.py` (output format, merge logic)
  - `src/edm/analysis/structure_detector.py` (1-indexing, event detection)
  - `src/cli/commands/analyze.py` (TrackAnalysis output formatting)
  - `src/edm/evaluation/evaluators/structure.py` (evaluation needs both formats)
  - Tests, fixtures, and example outputs

### 4. Raw Detection Output
- Add `raw` key containing all detected sections with full detail
- Includes: start/end times (seconds), start/end bars (fractional), label, confidence
- Preserves original detection before post-processing (merging, 1-indexing)
- Useful for debugging, analysis improvement, and understanding detector behavior

### 5. Annotation Template Output
- Add `--annotations` flag to analyze command
- When enabled, also outputs a simplified `.annotations.yaml` file alongside standard output
- Template format: just `[bar, label]` tuples for easy manual editing
- Includes all metadata (file, duration, bpm, downbeat, time_signature)
- User edits the template to provide ground truth annotations

## New Output Schema

```yaml
file: track.flac
duration: 242.2
bpm: 128.0
downbeat: 0.02
time_signature: 4/4
structure:
  - [1, 24, intro]
  - [25, 80, other]
  - [81, 92, breakdown]
  - [93, 104, buildup]
  - [105, 126, other]
  - [127, 130, outro]
events:
  - [25, drop]
  - [105, drop]
raw:
  - {start: 0.0, end: 45.2, start_bar: 0.0, end_bar: 24.1, label: intro, confidence: 0.9}
  - {start: 45.2, end: 60.5, start_bar: 24.1, end_bar: 32.3, label: drop, confidence: 0.85}
  - {start: 60.5, end: 90.0, start_bar: 32.3, end_bar: 48.0, label: other, confidence: 0.5}
  ...
```

### Annotation Template (with `--annotations` flag)

Saved as `track.annotations.yaml`:
```yaml
file: /path/to/track.flac
duration: 242.2
bpm: 128.0
downbeat: 0.02
time_signature: 4/4
annotations:
- [1, intro]
- [25, drop]
- [33, other]
- [81, breakdown]
- [93, buildup]
- [105, drop]
- [127, outro]
```

## Migration Notes

- Old format: `[0, 7, intro]` → New format: `[1, 8, intro]`
- Old format: `[32, drop]` (as span) → New format in events: `[33, drop]` (if 32 was 0-indexed, becomes 33 in 1-indexed)
- Existing annotations need bar numbers incremented by 1
