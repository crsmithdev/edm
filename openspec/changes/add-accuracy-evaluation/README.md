# Accuracy Evaluation Framework - Quick Reference

## Overview

This change adds an internal accuracy evaluation framework (script-based) to systematically test and validate the accuracy of EDM analysis algorithms (BPM detection, drop detection, etc.) against reference data. **This is developer tooling, not a public API.**

## Key Features

✅ **Library Module**: Proper `edm.evaluation` module with programmatic API
✅ **CLI Integration**: `edm evaluate` command for interactive use
✅ **Multiple Analysis Types**: BPM, drops (future), key (future) via subcommands
✅ **Flexible Sampling**: Random (with seed) or full dataset
✅ **Reference from Files**: CSV/JSON with hand-tagged data
✅ **Rich Metrics**: MAE, RMSE, accuracy rates, error distributions, outliers
✅ **Dual Output**: JSON (machine-readable) + Markdown (human-readable)
✅ **Optional Visualization**: Error distribution plots with matplotlib (graceful degradation if not installed)
✅ **Git Tracking**: Results include commit hash for version comparison
✅ **AI Assistant Optimized**: Easy for Claude to read, compare, and analyze

## Quick Start

### Basic Usage

```bash
# Evaluate BPM accuracy on 100 random files (CSV reference)
edm evaluate bpm \
  --source /path/to/music \
  --reference tests/fixtures/reference/bpm_tagged.csv

# Evaluate BPM accuracy using Spotify API
edm evaluate bpm \
  --source /path/to/music \
  --reference spotify

# Evaluate BPM accuracy using file metadata
edm evaluate bpm \
  --source /path/to/music \
  --reference metadata

# Evaluate full dataset
edm evaluate bpm \
  --source /path/to/music \
  --reference tests/fixtures/reference/bpm_tagged.csv \
  --full

# Reproducible sampling
edm evaluate bpm \
  --source /path/to/music \
  --reference tests/fixtures/reference/bpm_tagged.csv \
  --seed 42

# Custom sample size and tolerance
edm evaluate bpm \
  --source /path/to/music \
  --sample-size 50 \
  --tolerance 5.0 \
  --reference tests/fixtures/reference/bpm_tagged.csv
```

### Example Output

```
Discovered 1523 audio files
Sampling: 100 files (random, seed=42)
Loading reference: tests/fixtures/reference/bpm_tagged.csv
Evaluating files...
  [1/100] track001.mp3: Ref=128.0, Computed=127.8, Error=-0.2 ✓
  [2/100] track002.flac: Ref=140.0, Computed=139.5, Error=-0.5 ✓
  ...

Evaluation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Files: 100
Successful: 97
Failed: 3

Metrics:
  MAE: 1.84 BPM
  RMSE: 2.91 BPM
  Accuracy (±2.5 BPM): 87.6%

Results saved to:
  - benchmarks/results/accuracy/bpm/2025-11-24_bpm_eval_commit-abc123.json
  - benchmarks/results/accuracy/bpm/2025-11-24_bpm_eval_commit-abc123.md
  - benchmarks/results/accuracy/bpm/2025-11-24_bpm_eval_commit-abc123.png (optional)
  - benchmarks/results/accuracy/bpm/latest.json (symlink)
  - benchmarks/results/accuracy/bpm/latest.md (symlink)
```

## Architecture

```
src/edm/
└── evaluation/              # Library module
    ├── __init__.py          # Public API
    ├── common.py            # Discovery, sampling, metrics, I/O
    ├── reference.py         # Reference loading
    └── evaluators/
        ├── __init__.py
        ├── bpm.py           # BPM evaluation
        └── drops.py         # Drop evaluation (future)

src/cli/commands/
└── evaluate.py              # CLI: edm evaluate

tests/
├── fixtures/reference/      # Reference data
│   ├── bpm_tagged.csv
│   └── README.md
└── unit/test_evaluation/    # Tests
    ├── test_common.py
    ├── test_reference.py
    └── test_bpm.py

benchmarks/results/          # Results output
└── accuracy/
    └── bpm/
        ├── 2025-11-24_bpm_eval_commit-abc123.json
        ├── 2025-11-24_bpm_eval_commit-abc123.md
        ├── latest.json
        └── latest.md
```

## Reference Sources

### Spotify API (`--reference spotify`)
- **BPM only** - Automatically looks up BPM for discovered files
- Requires Spotify API credentials configured
- Best for initial evaluation without manual tagging
- Results are cached to avoid repeated API calls

```bash
python benchmarks/accuracy/evaluate.py bpm \
  --source /path/to/music \
  --reference spotify
```

### File Metadata (`--reference metadata`)
- **BPM, key** - Reads directly from file metadata (ID3/Vorbis/MP4 tags)
- No external API calls or credentials needed
- Best when files come from authoritative sources (Beatport, Rekordbox, Traktor, etc.)
- Uses existing `edm.io.metadata.read_metadata()` functionality

```bash
python benchmarks/accuracy/evaluate.py bpm \
  --source /path/to/music \
  --reference metadata
```

### CSV Format (`--reference path/to/file.csv`)
```csv
path,bpm
/music/track1.mp3,128.0
/music/track2.flac,140.0
/music/track3.wav,174.0
```

### JSON Format (`--reference path/to/file.json`)
```json
[
  {"path": "/music/track1.mp3", "bpm": 128.0},
  {"path": "/music/track2.flac", "bpm": 140.0},
  {"path": "/music/track3.wav", "bpm": 174.0}
]
```

**Note**: Drop detection only supports CSV/JSON (no API or metadata source available). Key detection supports metadata, CSV, or JSON.

## Results Format

### Markdown (`latest.md`)
```markdown
# BPM Evaluation Results

**Date**: 2025-11-24 14:30:52  
**Commit**: abc123def456  
**Sample**: 100 files (random, seed=42)  

## Summary Metrics
- MAE: 1.84 BPM
- RMSE: 2.91 BPM
- Accuracy (±2.5 BPM): 87.6%

## Worst Outliers
| File | Reference | Computed | Error |
|------|-----------|----------|-------|
| track1.mp3 | 128.0 | 85.3 | -42.7 |
```

### JSON (`latest.json`)
```json
{
  "metadata": {
    "analysis_type": "bpm",
    "timestamp": "2025-11-24T14:30:52",
    "git_commit": "abc123def456",
    "git_branch": "main",
    "sample_size": 100,
    "sampling_strategy": "random",
    "sampling_seed": 42,
    "tolerance": 2.5
  },
  "summary": {
    "total_files": 100,
    "successful": 97,
    "failed": 3,
    "mean_absolute_error": 1.84,
    "root_mean_square_error": 2.91,
    "accuracy_within_tolerance": 87.6
  },
  "results": [...]
}
```

## AI Assistant Integration

This format is optimized for AI assistants like Claude:

**Quick queries:**
- "What's the current BPM accuracy?" → Read `latest.md`
- "Compare accuracy across commits" → Compare JSON files by git_commit
- "Show worst outliers" → Read outliers section in Markdown
- "Has accuracy improved?" → Compare MAE/RMSE across evaluations

**Example conversation:**
```
User: "Did my BPM algorithm changes improve accuracy?"
Claude: *Reads latest.md and previous evaluation*
"Yes! MAE improved from 2.14 BPM to 1.84 BPM (14% better).
Accuracy within ±2.5 BPM increased from 82.3% to 87.6%."
```

## Implementation Phases

1. **Phase 1**: Core utilities (`src/edm/evaluation/common.py`)
2. **Phase 2**: Reference loading (`src/edm/evaluation/reference.py`)
3. **Phase 3**: BPM evaluator (`src/edm/evaluation/evaluators/bpm.py`)
4. **Phase 4**: CLI command (`src/cli/commands/evaluate.py`)
5. **Phase 5**: Reference fixtures setup
6. **Phase 6**: Results documentation
7. **Phase 7**: Testing
8. **Phase 8**: Documentation
9. **Phase 9**: QA

**Estimated time**: 10-15 hours

## Why Library Module?

- ✅ **Proper architecture**: Follows project structure
- ✅ **Reusable**: Can import in notebooks/scripts
- ✅ **Testable**: Standard pytest conventions
- ✅ **Maintainable**: ~450 LOC core logic + 100 LOC CLI
- ✅ **Still internal**: Not part of public API docs
- ✅ **Flexible**: CLI and programmatic access

## Next Steps

1. Review proposal documents
2. Create initial reference dataset
3. Implement according to `tasks.md`
4. Test with real music files
5. Use for algorithm development and validation

## Documentation

- **proposal.md**: Full proposal with motivation and solution
- **design.md**: Technical design details (if needed)
- **tasks.md**: Implementation checklist (58 tasks across 9 phases)
- **specs/**: Specification deltas for scripts and core library

## Questions?

This is internal tooling for developers and maintainers. If you have questions about usage or implementation, refer to the full proposal or ask!
