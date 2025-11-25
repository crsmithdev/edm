# Change Proposal: Add Accuracy Evaluation Framework

**Change ID**: `add-accuracy-evaluation`  
**Status**: Draft  
**Created**: 2025-11-24  
**Author**: AI Assistant

## Summary

Add an internal accuracy evaluation framework to systematically test and validate the accuracy of EDM analysis algorithms (BPM detection, drop detection, etc.) against reference data from multiple sources. This is a developer/maintainer tool, not intended for public use.

## Motivation

Currently, there is no systematic way to:
- Evaluate the accuracy of BPM detection or other analysis algorithms
- Compare computed results against reference from metadata or external APIs
- Test algorithm performance on random samples of music files
- Store and track evaluation results over time
- Validate improvements or regressions in algorithm accuracy

This makes it difficult to:
- Confidently deploy algorithm improvements
- Identify edge cases or problematic track types
- Benchmark against external data sources (Spotify, manual tags)
- Track accuracy metrics over time
- Compare results across commits/versions

## Goals
3. **Configurable Sampling**: Random, stratified, or full-dataset sampling
4. **Multiple Reference Sources**:
   - Hand-tagged datasets (CSV/JSON)
   - Local file metadata (ID3 tags)
   - Spotify API (optional)
5. **Results Persistence**: Store results in human-readable and machine-parseable format

## Solution

### Architecture

Implement as a proper library module with CLI integration:

```
src/edm/
└── evaluation/              # New library module
    ├── __init__.py          # Public API: evaluate_bpm(), evaluate_drops()
    ├── common.py            # Discovery, sampling, metrics, I/O
    ├── reference.py         # Reference loading (spotify, metadata, files)
    └── evaluators/
        ├── __init__.py
        ├── bpm.py           # BPM evaluation logic
        ├── drops.py         # Drop detection evaluation (future)
        └── key.py           # Key detection evaluation (future)

src/cli/commands/
└── evaluate.py              # CLI command: edm evaluate

tests/
├── fixtures/reference/      # Reference data
│   ├── bpm_tagged.csv
│   ├── drops_tagged.json
│   └── README.md
└── unit/
    └── test_evaluation/     # Tests for edm.evaluation module
        ├── test_common.py
        ├── test_reference.py
        └── test_bpm.py

benchmarks/results/          # Evaluation results output
└── accuracy/
    ├── bpm/
    │   ├── 2025-11-24_bpm_eval_commit-abc123.json
    │   ├── 2025-11-24_bpm_eval_commit-abc123.md
    │   └── latest.json
    └── README.md
```

**Key Benefits:**
- ✅ Proper library code under `src/edm/evaluation/` (~450 LOC core logic)
- ✅ CLI integration via `edm evaluate` command (~100 LOC)
- ✅ Reusable - import `from edm.evaluation import evaluate_bpm()` in notebooks/scripts
- ✅ Testable - standard pytest structure
- ✅ Still internal - not part of public API docs
- ✅ Maintainable - follows project conventions

### CLI Interface

```bash
# Evaluate BPM accuracy with CSV reference
edm evaluate bpm \
  --source /path/to/music \
  --sample-size 100 \
  --reference tests/fixtures/reference/bpm_tagged.csv

# Evaluate BPM accuracy using Spotify API
edm evaluate bpm \
  --source /path/to/music \
  --reference spotify

# Evaluate BPM accuracy using file metadata (ID3 tags)
edm evaluate bpm \
  --source /path/to/music \
  --reference metadata

# Evaluate with all files
edm evaluate bpm \
  --source /path/to/music \
  --reference tests/fixtures/reference/bpm_tagged.csv \
  --full

# Reproducible sampling
edm evaluate bpm \
  --source /path/to/music \
  --reference metadata \
  --seed 42

# Future: Evaluate drops (Spotify not supported)
edm evaluate drops \
  --source /path/to/music \
  --reference tests/fixtures/reference/drops_tagged.json
```

### Programmatic Usage

```python
from edm.evaluation import evaluate_bpm
from pathlib import Path

# Use in scripts or notebooks
results = evaluate_bpm(
    source_path=Path("/path/to/music"),
    reference_source="spotify",
    sample_size=100,
    seed=42
)

print(f"MAE: {results['summary']['mean_absolute_error']:.2f} BPM")
```

## Dependencies

- Existing: `mutagen`, `structlog`
- New: None required
- Optional: `matplotlib` for visualization (~30 lines with graceful degradation)

## Key Features

1. **Core Utilities** (`common.py`):
   - File discovery and sampling (random with seed, full dataset)
   - Metrics calculation (MAE, RMSE, accuracy, octave-aware, tempo-stratified)
   - Result storage (JSON + Markdown)
   - Path normalization for WSL compatibility

2. **Reference Loading** (`reference.py`):
   - Unified `load_reference()` with auto-detection
   - Spotify API integration (BPM only, with permanent caching)
   - File metadata reading (ID3/Vorbis/MP4 tags for BPM, key)
   - CSV/JSON file loading
   - Analysis-specific validation

3. **Evaluators** (`evaluators/`):
   - `bpm.py`: BPM evaluation using `edm.analysis.bpm`
   - `drops.py`: Drop detection evaluation (future)
   - `key.py`: Key detection evaluation (future)

4. **CLI Command** (`src/cli/commands/evaluate.py`):
   - `edm evaluate bpm` - BPM accuracy evaluation
   - `edm evaluate drops` - Drop detection (future)
   - `edm evaluate key` - Key detection (future)

5. **Reference Sources** (via unified `--reference` argument):
   - **Spotify API**: `--reference spotify` (BPM only)
     - Automatically looks up BPM for discovered files using track metadata
     - Requires Spotify API credentials configured in `edm.external.spotify`
     - Best for initial evaluation without manual tagging
   - **File metadata (ID3 tags)**: `--reference metadata` (BPM, key)
     - Reads BPM/key directly from file metadata (ID3/Vorbis/MP4 tags)
     - Uses existing `edm.io.metadata.read_metadata()` functionality
     - Best when files come from authoritative sources (Beatport, Rekordbox, etc.)
     - No external API calls or manual tagging required
   - **CSV files**: `--reference path/to/file.csv`
     - Format: `path,bpm` (or other value fields for different analyses)
     - Best for hand-tagged reference data
   - **JSON files**: `--reference path/to/file.json`
     - Format: `[{"path": "...", "bpm": 128.0}, ...]`

6. **Metrics**:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Accuracy within tolerance (% correct within ±X BPM)
   - **Octave-aware accuracy** (detects halving/doubling errors common in BPM detection)
   - **Tempo-stratified breakdown** (accuracy by tempo range: slow/medium/fast/very_fast)
   - Error distribution histogram
   - Worst N outliers with file paths
   - Confidence score tracking (if available from algorithm)

7. **Results Format** (optimized for AI assistant parsing):
   - Human-readable JSON with clear structure
   - Git commit hash for version tracking
   - Timestamp and configuration snapshot
   - Individual file results for debugging
   - Summary statistics prominently at top
   - Markdown summary for quick reading

### Example Output Format

**Human-readable summary** (`benchmarks/results/accuracy/bpm/latest.md`):
```markdown
# BPM Evaluation Results

**Date**: 2025-11-24 14:30:52
**Commit**: abc123def456
**Sample Size**: 100 files (random sampling, seed=42)
**Reference**: tests/fixtures/reference/bpm_tagged.csv

## Summary Metrics

- **Mean Absolute Error (MAE)**: 1.84 BPM
- **Root Mean Square Error (RMSE)**: 2.91 BPM
- **Accuracy (±2.5 BPM)**: 87.6%
- **Successful Evaluations**: 97 / 100
- **Failed Evaluations**: 3

## Worst Outliers

| File | Reference | Computed | Error |
|------|-----------|----------|-------|
| track1.mp3 | 128.0 | 85.3 | -42.7 |
| track2.flac | 140.0 | 70.1 | -69.9 |
| track3.wav | 174.0 | 87.0 | -87.0 |

## Error Distribution

- [-10, -5): 2 files
- [-5, 0): 12 files
- [0, 5): 78 files
- [5, 10): 5 files
- [10+): 3 files
```

**Machine-readable details** (`benchmarks/results/accuracy/bpm/2025-11-24_bpm_eval_commit-abc123.json`):
```json
{
  "metadata": {
    "analysis_type": "bpm",
    "algorithm_version": "madmom-0.16.1",
    "timestamp": "2025-11-24T14:30:52.123456",
    "git_commit": "abc123def456",
    "git_branch": "main",
    "sample_size": 100,
    "sampling_strategy": "random",
    "sampling_seed": 42,
    "reference_source": "tests/fixtures/reference/bpm_tagged.csv",
    "tolerance": 2.5
  },
  "summary": {
    "total_files": 100,
    "successful": 97,
    "failed": 3,
    "mean_absolute_error": 1.84,
    "root_mean_square_error": 2.91,
    "octave_errors": 5,
    "accuracy_with_octave_correction": 92.8,
    "tempo_stratified": {
      "slow_100_120": {"count": 15, "mae": 1.2, "accuracy": 93.3},
      "medium_120_135": {"count": 60, "mae": 1.5, "accuracy": 91.7},
      "fast_135_150": {"count": 20, "mae": 2.8, "accuracy": 75.0},
      "very_fast_150plus": {"count": 5, "mae": 4.1, "accuracy": 60.0}
    },
    "accuracy_within_tolerance": 87.6,
    "error_distribution": {
      "[-10, -5)": 2,
      "[-5, 0)": 12,
      "[0, 5)": 78,
      "[5, 10)": 5,
      "[10+)": 3
    }
  },
  "outliers": [
    {
      "file": "track1.mp3",
      "reference": 128.0,
      "computed": 85.3,
      "error": -42.7,
      "is_octave_error": false,
      "confidence": 0.87,
      "error_message": null
    }
  ],
  "results": [
    {
      "file": "track1.mp3",
      "reference": 128.0,
      "computed": 127.8,
      "error": -0.2,
      "is_octave_error": false,
      "confidence": 0.95,
      "success": true,
      "computation_time": 0.843
    }
  ]
}
```

**Why this format?**
- Markdown summary is easy for Claude to read and compare
- JSON has full details for programmatic analysis
- Git commit tracking enables comparison: "How does accuracy on commit abc123 compare to def456?"
- Clear separation of metadata, summary, outliers, and individual results
- Symlink to `latest.json` and `latest.md` makes it easy to find most recent results

## Success Criteria

- [ ] Can evaluate BPM accuracy on random sample or full dataset
- [ ] Can load reference from CSV/JSON files, Spotify API, and file metadata
- [ ] Results stored in both JSON (machine-readable) and Markdown (human-readable)
- [ ] Git commit hash tracked in results for version comparison
- [ ] Script is simple to run: `python benchmarks/accuracy/evaluate.py bpm --source ~/music --reference tests/fixtures/reference/bpm_tagged.csv`
- [ ] Results format is easy for AI assistants to parse and compare
- [ ] Framework supports multiple analysis types through subcommands
- [ ] Comprehensive test coverage for accuracy module

## Non-Goals

- **Public API or CLI command** - This is internal tooling only
- Real-time evaluation (batch processing is sufficient)
- GUI or web interface
- Automatic algorithm tuning (just measurement)
- Database storage (JSON + Markdown files are sufficient)
- Spotify API as primary reference (prefer hand-tagged data for production use)

## Impact Assessment

### Benefits
- Systematic validation of algorithm accuracy
- Faster iteration on algorithm improvements
- Better understanding of edge cases
- Historical tracking of accuracy over time

### Risks
- Reference data quality varies - need careful manual tagging
- Large sample sizes may take time - show progress output
- Maintaining reference datasets requires manual effort

### Alternatives Considered

1. **Full CLI integration** (`edm evaluate`): Rejected - overkill for internal tool, adds complexity
2. **Jupyter notebooks only**: Considered but scripts are better for reproducibility and CI/CD
3. **pytest-based approach**: Considered but separate scripts are more flexible for ad-hoc evaluation
4. **Separate per-analysis scripts**: Rejected - too much duplication, harder to maintain with multiple analysis types

## Dependencies

- Existing: `mutagen`, `structlog` (for logging)
- New: None required (all standard library for scripts)
- Optional: `matplotlib` for error distribution visualization (~30 lines of code with graceful degradation)

## Timeline Estimate

- Shared infrastructure: 2-3 hours
- BPM evaluator: 2-3 hours
- Reference handling: 1-2 hours
- Testing: 2-3 hours
- Documentation and reference setup: 1-2 hours
- **Total**: ~8-13 hours (simpler than full CLI integration)

## Reference Source Design

### Unified `--reference` Argument

The `--reference` argument accepts multiple input types with automatic detection:

```python
# benchmarks/accuracy/common.py
def load_reference_auto(reference_arg: str, analysis_type: str, source_path: Path,
                       value_field: str = "bpm") -> Dict[Path, Any]:
    """Auto-detect and load reference based on argument and analysis type."""
    if reference_arg.lower() == "spotify":
        if analysis_type not in ["bpm"]:
            raise ValueError(f"Spotify reference not supported for {analysis_type} analysis")
        return load_spotify_reference(source_path)

    if reference_arg.lower() == "metadata":
        if analysis_type not in ["bpm", "key"]:
            raise ValueError(f"Metadata reference not supported for {analysis_type} analysis")
        return load_metadata_reference(source_path, value_field)

    ref_path = Path(reference_arg)
    if ref_path.suffix == ".csv":
        return load_reference_csv(ref_path)
    elif ref_path.suffix == ".json":
        return load_reference_json(ref_path)
    else:
        raise ValueError(f"Unknown reference format: {reference_arg}")

def load_spotify_reference(source_path: Path) -> Dict[Path, float]:
    """Load BPM data from Spotify API for discovered files."""
    from edm.external.spotify import SpotifyClient
    from edm.io.metadata import read_metadata

    client = SpotifyClient()
    reference = {}

    files = discover_audio_files(source_path)
    for file_path in files:
        try:
            metadata = read_metadata(file_path)
            track_info = client.search_track(
                metadata.get("artist"),
                metadata.get("title")
            )
            if track_info and track_info.get("bpm"):
                reference[file_path] = float(track_info["bpm"])
        except Exception as e:
            logger.warning("spotify_lookup_failed", file=str(file_path), error=str(e))

    return reference

def load_metadata_reference(source_path: Path, value_field: str = "bpm") -> Dict[Path, Any]:
    """Load reference data from file metadata (ID3/Vorbis/MP4 tags)."""
    from edm.io.metadata import read_metadata

    reference = {}
    files = discover_audio_files(source_path)

    for file_path in files:
        try:
            metadata = read_metadata(file_path)
            if value_field in metadata and metadata[value_field]:
                # Convert to appropriate type based on field
                if value_field == "bpm":
                    reference[file_path] = float(metadata[value_field])
                elif value_field == "key":
                    reference[file_path] = str(metadata[value_field])
                else:
                    reference[file_path] = metadata[value_field]
        except Exception as e:
            logger.warning("metadata_read_failed", file=str(file_path),
                         field=value_field, error=str(e))

    return reference
```

**Benefits:**
- Clean interface: one argument instead of multiple mutually exclusive flags
- Extensible: easy to add more reference types (e.g., `--reference beatport`)
- Analysis-specific: each analysis declares what sources it supports
- Clear error messages when unsupported combinations are used

**Spotify API Caching:**
- Results permanently cached to avoid repeated API calls (BPM for a track doesn't change)
- Cache stored in `benchmarks/results/accuracy/.cache/spotify_bpm.json`
- Format: `{"file_path": {"bpm": 128.0, "track_id": "spotify:track:...", "fetched_at": "2025-11-24"}}`
- Cache is treated as reference data and can be committed to git
- Can be manually edited if Spotify data is incorrect

**Path Normalization:**
- Reference file paths are normalized relative to `--source` directory
- Handles WSL path translation (`/mnt/c/...` vs `C:\...`)
- Uses `Path.resolve()` and `Path.relative_to()` for consistent matching
- Fallback to filename-based matching if paths don't share common root

## Questions for Review

1. Should we create initial reference dataset, or wait for manual tagging?
2. Should Spotify API cache results permanently or use time-based expiration?
3. Should we add visualization (matplotlib plots) of error distributions?
4. Should results be stored in git for historical tracking?

## References

- Similar tools: MLflow, Weights & Biases (inspiration for metrics tracking)
- Industry standard BPM tolerance: ±5% (DJ software)
- Spotify API audio features: https://developer.spotify.com/documentation/web-api/reference/get-audio-features

## AI Assistant Integration

The results format is specifically designed to be easy for AI assistants (like Claude) to:

1. **Quick Summary**: Read `results/accuracy/bpm/latest.md` for immediate understanding
2. **Compare Across Commits**: "Compare BPM accuracy between commit abc123 and def456"
3. **Identify Regressions**: "Has accuracy gotten worse since last week?"
4. **Debug Outliers**: "Show me the files with largest errors in latest evaluation"
5. **Track Improvements**: "How has MAE changed over the last 5 evaluations?"

Examples of queries AI can answer:
- "What's the current BPM accuracy?"
- "Which files are consistently problematic?"
- "Did my changes improve accuracy?"
- "Show me error distribution for the latest run"

The combination of Markdown (human-readable) + JSON (machine-parseable) + git commit tracking makes this seamless.

**Example - Bulk outlier review:**
```
User: "Show me the worst 10 BPM outliers from the latest evaluation"
Claude: *Reads benchmarks/results/accuracy/bpm/latest.json, shows outliers table*
User: "Fix #3 and #7 - actual BPMs are 128 and 174"
Claude: *Updates tests/fixtures/reference/bpm_tagged.csv with corrections*
```

### Interactive Hand-Tagging Workflow

For capturing manual corrections during analysis, use direct conversation:

**Example - BPM correction:**
```
User: "The BPM for track.mp3 is actually 140, not 128"
Claude: *Appends to tests/fixtures/reference/bpm_tagged.csv*
        /path/to/track.mp3,140.0
```

**Example - Drop detection correction:**
```
User: "Track xyz.flac has drops at 1:23.5, 2:47.2, and 4:15.8"
Claude: *Appends to tests/fixtures/reference/drops_tagged.json*
        {
          "path": "/path/to/xyz.flac",
          "drops": [83.5, 167.2, 255.8],
          "tagged_by": "human",
          "tagged_date": "2025-11-24",
          "reason": "Outlier from evaluation abc123 - detected 2 drops, actual has 3",
          "confidence": "high"
        }
```

**Benefits:**
- No special tooling needed - just natural conversation
- Reference data persists in CSV/JSON files committed to git
- Can review and correct in batches or one-at-a-time
- Historical record of what was tagged and when (via git commits)
