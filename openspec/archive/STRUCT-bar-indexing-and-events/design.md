# Design: Bar Indexing and Event Format

## Problem Analysis

### Current State
```yaml
# Current output (0-indexed, events mixed with spans)
structure:
- [1, 1, intro]      # confusing: 1 bar at bar 1?
- [1, 16, other]     # overlaps with previous
- [24, drop]         # event as span with same start/end
- [32, drop]
```

### Issues
1. Bar 1 in code = bar 1 in output, but span lengths are confusing (end-start doesn't equal duration)
2. Events (drops) have same start/end bar, semantically incorrect
3. Multiple 'other' sections clutter output

## Solution Design

### 1. Bar Indexing Convention

**Current (0-indexed style but inconsistent):**
- Internal: `start_bar=0.0, end_bar=8.0` (8 bars)
- Output: `[1, 8, intro]` but next starts at 9 (implies inclusive end)

**New (1-indexed, inclusive ranges):**
- Internal: Keep float calculations as-is for beat grid
- Output conversion: `[ceil(start_bar)+1, ceil(end_bar), label]`
- Interpretation: Bars 1-8 inclusive = 8 bars, next section starts at bar 9

**Edge cases:**
- Fractional bars: Round start up, end down for conservative ranges
- Zero-duration: Skip in output
- First section: Always starts at bar 1

### 2. Event vs Span Separation

**Detection logic changes:**

In `structure_detector.py`:
```python
@dataclass
class DetectedSection:
    start_time: float
    end_time: float
    label: str
    confidence: float
    is_event: bool = False  # NEW FIELD
```

**Event identification rules:**
- `drop` → always event (mark start time only)
- `kick` → always event (percussion onset)
- All other labels → spans

**Output formatting in `structure.py`:**
```python
def format_structure_output(sections, bpm, time_signature, downbeat):
    spans = []
    events = []

    for section in sections:
        if section.is_event:
            bar = time_to_bars(section.start_time, bpm, time_signature, downbeat)
            events.append([ceil(bar) + 1, section.label])  # 1-indexed
        else:
            start_bar = time_to_bars(section.start_time, bpm, time_signature, downbeat)
            end_bar = time_to_bars(section.end_time, bpm, time_signature, downbeat)
            spans.append([ceil(start_bar) + 1, floor(end_bar), section.label])

    return spans, events
```

### 3. Merge Consecutive 'other' Sections

**Post-processing in `structure.py`:**
```python
def merge_consecutive_other(sections: list[Section]) -> list[Section]:
    """Merge consecutive sections with label='other'."""
    if not sections:
        return []

    merged = [sections[0]]

    for section in sections[1:]:
        prev = merged[-1]

        if section.label == 'other' and prev.label == 'other':
            # Merge: extend previous section's end time
            merged[-1] = Section(
                label='other',
                start_time=prev.start_time,
                end_time=section.end_time,
                confidence=max(prev.confidence, section.confidence),
                start_bar=prev.start_bar,
                end_bar=section.end_bar,
                bar_count=None,  # Recalculate if needed
            )
        else:
            merged.append(section)

    return merged
```

**Apply after post-processing, before bar calculations.**

## Implementation Plan

### Phase 1: Add Event Support
1. Add `is_event` field to `DetectedSection`
2. Update detectors to mark drops as events
3. Add `kick` event detection (percussion onset via librosa)
4. Split output into `structure` and `events` keys

### Phase 2: Fix Bar Indexing
1. Update bar calculation output to use 1-indexing
2. Update all tests and fixtures
3. Document convention clearly

### Phase 3: Merge Consecutive 'other'
1. Add merge function
2. Apply in post-processing pipeline
3. Test edge cases (all 'other', first/last section, etc.)

## Testing Strategy

### Unit Tests
```python
def test_bar_indexing_one_indexed():
    # 8-bar intro at start should be [1, 8, intro]
    section = Section(label='intro', start_time=0.0, end_time=15.0, ...)  # 15s at 128 BPM = 8 bars
    result = format_section(section, bpm=128, ...)
    assert result == [1, 8, 'intro']

def test_event_format():
    drop = DetectedSection(start_time=60.0, label='drop', is_event=True, ...)
    spans, events = format_structure_output([drop], bpm=128, ...)
    assert events == [[33, 'drop']]  # Bar 33 at 60s with 128 BPM
    assert spans == []

def test_merge_consecutive_other():
    sections = [
        Section(label='other', start_time=0, end_time=30, ...),
        Section(label='other', start_time=30, end_time=60, ...),
        Section(label='drop', start_time=60, end_time=90, ...),
    ]
    merged = merge_consecutive_other(sections)
    assert len(merged) == 2
    assert merged[0].label == 'other'
    assert merged[0].end_time == 60
```

### Integration Tests
- Compare output against annotated track (`dj-rhythmcore-narcos_annotated.yaml`)
- Verify bar numbers match DJ software display

## Migration

### Existing Annotations
All existing YAML annotations with 0-indexed bars need +1 offset:
```bash
# Script to migrate annotations
for file in data/**/*.yaml; do
    # Increment all bar numbers by 1
    sed -i 's/\[\([0-9]*\), \([0-9]*\), /\[$((\1+1)), $((\2+1)), /' $file
done
```

### Documentation Updates
- Update `STRUCTURE_ANNOTATIONS_GUIDE.md`
- Update CLI help text
- Add migration notes to CHANGELOG
