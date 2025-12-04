# [WORKFLOW]Restructure Annotation Workflow

**Status:** deployed
**Created:** 2025-12-03
**Updated:** 2025-12-03
**Archived:** 2025-12-03

## Why

Current annotation workflow has fragmented concerns across three slash commands (`/annotate generate`, `/annotate merge`, `/annotate save`) with three separate directories (reference/, generated/, working/). This creates:

1. **Cognitive overhead**: Users must understand merge/save concepts and remember which directory contains what
2. **Context fragmentation**: Audio analysis knowledge split between `/annotate` and `/evaluate` contexts
3. **Limited raw data visibility**: Only MSAF segments preserved; no per-detector breakdown
4. **Indirect workflow**: Must run `edm analyze` then merge/save instead of direct annotation update

The system should support a simpler mental model: analyze → annotate → evaluate, with one command per step and unified context knowledge.

## What

Restructure workflow into three clean commands matching the mental model:

### Commands

**`/analyze [files]`**: Analyze audio files, save YAML to `data/generated/`
- Calls `edm analyze --annotations` with structured YAML output
- Preserves all raw detector data (MSAF, energy-based, future detectors)
- Outputs to `data/generated/` (git-ignored, disposable)

**`/annotate`**: Merge generated data into `data/annotations/`
- For each file in `data/generated/`:
  - If exists in `data/annotations/`: update with new raw data (commented YAML block)
  - If new: create annotation file with raw data as comments
- Never deletes user data from `data/annotations/`
- Output: `data/annotations/` (git-tracked, authoritative)

**`/evaluate [files]`**: Compare analysis accuracy
- With args: evaluate specific tracks from `data/generated/` vs `data/annotations/`
- Without args: evaluate all tracks present in both directories
- Reports F-scores, precision, recall per section type

### Context Consolidation

Merge current `annotate-context` and `evaluate-context` into single `evaluation-context` containing:
- CLI usage: `edm analyze`, flags, structured output format
- Directories: `data/generated/` (analysis), `data/annotations/` (ground truth), `~/music` (test files)
- Safety: NEVER delete from `data/annotations/`, merge strategy
- Evaluation: F-scores, boundary tolerance (±2s), metrics definitions
- Analysis: BPM cascade, structure detectors (MSAF/energy), beat_this integration

### Raw Data Enhancement

Analysis output includes per-detector raw data:

```yaml
file: "/home/crsmi/music/track.flac"
duration: 209.1
bpm: 132.0
downbeat: 0.92
time_signature: 4/4
structure:
- [2, 7, segment5]
- [7, 25, segment2]
raw:
  msaf:
  - {start: 0.0, end: 10.26, start_bar: 1, end_bar: 6, label: segment5, confidence: 0.8}
  energy:
  - {start: 0.0, end: 11.5, start_bar: 1, end_bar: 7, label: intro, confidence: 0.6}
```

Annotation files store raw data as commented YAML after `---` separator (existing behavior preserved).

## Impact

**Breaking changes:**
- `/annotate` command signature changes: no more `generate/merge/save` subcommands
- `/analyze` slash command replaces `/annotate generate`
- Context files: `.claude/contexts/annotate.xml` + `.claude/contexts/evaluate.xml` → `.claude/contexts/evaluation.xml`

**Affected specs:**
- `openspec/specs/cli/spec.md`: Add `/analyze` command requirements
- `openspec/specs/analysis/spec.md`: Update raw data format requirements

**Dependencies:**
- `edm analyze` must support `--annotations` output format (verify current state)
- YAML output must include per-detector raw sections
