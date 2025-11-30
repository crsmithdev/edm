# Change: Polymorphic Structure Format

## Why
Drops are singular moments (the bar where the beat kicks in), not spans with start/end. The current format treats all structure elements as spans, which misrepresents drops and will misrepresent future event-type elements.

## What Changes
- **BREAKING**: Structure output uses polymorphic tuples:
  - Spans: `[start_bar, end_bar, label]` (3 elements)
  - Events: `[bar, label]` (2 elements)
- Update structure detector to identify drop moments vs sections
- Update evaluation logic to handle both tuple types
- Initial event labels: `drop`
- Initial span labels: `intro`, `outro`, `breakdown`, `buildup`, `verse`, `chorus`, `bridge`, `other`

## Impact
- Affected specs: `analysis`
- Affected code:
  - `src/edm/analysis/structure.py` (output format)
  - `src/edm/analysis/structure_detector.py` (detection logic)
  - `src/cli/commands/analyze.py` (TrackAnalysis dataclass)
  - `src/edm/evaluation/evaluators/structure.py` (evaluation)
  - Tests and fixtures

## New Output Schema

```yaml
file: track.flac
duration: 342.5
tempo:
  bpm: 128.0
structure:
  - [1, 16, intro]       # span
  - [17, drop]           # event
  - [17, 48, high_energy] # span (optional, post-drop section)
  - [49, 64, breakdown]  # span
  - [65, drop]           # event
  - [65, 96, high_energy]
  - [97, 100, outro]     # span
```

## Implementation
- When current detector labels a section as "drop", emit only the start bar as an event
- No new detection logic needed - just a formatting change at output time
- Event labels: `drop` (future: other moment-based labels)
- Span labels: everything else (`intro`, `breakdown`, `buildup`, etc.)
