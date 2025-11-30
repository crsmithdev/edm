# Change: Add YAML Output Format

## Why
The current JSON output is flat and mixes metadata with analysis results. A cleaner, hierarchical format improves human readability for annotation workflows where users correct machine-computed values.

## What Changes
- Add `--format yaml` option to `edm analyze`
- Restructure output schema for both JSON and YAML:
  - Group related fields (`tempo.bpm`, `tempo.downbeat`, `tempo.time_signature`)
  - Use bar-based structure sections `[start_bar, end_bar, label]` for compactness
  - Remove redundant fields (`sections` count when `structure` list exists)
- Multi-document YAML for batch output (streaming-friendly)
- **BREAKING**: JSON output structure changes to match new schema

## Impact
- Affected specs: `cli`
- Affected code: `src/cli/commands/analyze.py`

## New Output Schema

### Single track:
```yaml
file: "track.flac"
duration: 342.5

tempo:
  bpm: 130.4
  time_signature: 4/4
  downbeat: 0.231

key: Am

structure:
  - [1, 4, intro]
  - [5, 36, drop]
  - [37, 52, breakdown]
  - [53, 84, drop]
  - [85, 96, outro]
```

### Batch (multi-document YAML):
```yaml
file: "track1.flac"
duration: 342.5
tempo:
  bpm: 130.4
  time_signature: 4/4
  downbeat: 0.231
key: Am
structure:
  - [1, 4, intro]
---
file: "track2.flac"
duration: 289.1
tempo:
  bpm: 128.0
  time_signature: 4/4
  downbeat: 0.102
key: Fm
structure:
  - [1, 8, intro]
```

## Design Rationale
- **Bar-based sections**: EDM tracks are grid-aligned; bars are more intuitive than timestamps
- **Multi-document YAML**: Streaming-friendly, each track is self-contained, easy to split/append
- **Compact tuple format**: `[start, end, label]` is readable and concise for hand-editing
- **Grouped tempo fields**: `bpm`, `downbeat`, `time_signature` belong together conceptually
