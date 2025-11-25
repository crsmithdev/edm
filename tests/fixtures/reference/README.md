# Reference Data for Accuracy Evaluation

This directory contains reference (ground truth) data for evaluating the accuracy of EDM analysis algorithms.

## File Formats

### CSV Format

Used for simple single-value references (BPM, key, etc.):

```csv
path,bpm
/path/to/track1.mp3,128.0
/path/to/track2.flac,140.0
/path/to/track3.wav,174.0
```

**Columns:**
- `path`: Absolute or relative path to audio file
- `bpm` (or other value field): Reference value

### JSON Format

Used for complex multi-value references (drop timestamps, etc.):

```json
[
  {
    "path": "/path/to/track1.mp3",
    "bpm": 128.0
  },
  {
    "path": "/path/to/track2.flac",
    "drops": [83.5, 167.2, 255.8],
    "tagged_by": "human",
    "confidence": "high"
  }
]
```

**Fields:**
- `path`: Audio file path (required)
- Value fields vary by analysis type (bpm, drops, key, etc.)
- Optional metadata: tagged_by, confidence, notes

## Creating Reference Data

### Manual Tagging

1. Listen to tracks and identify reference values
2. Add entries to CSV or JSON file
3. Use absolute paths or paths relative to evaluation source directory

### From Spotify API

Use the evaluation tool with `--reference spotify` to automatically fetch BPM data:

```bash
edm evaluate bpm --source ~/music --reference spotify
```

Results are cached and can be saved as reference data.

### From File Metadata

Use the evaluation tool with `--reference metadata` to extract from ID3/Vorbis tags:

```bash
edm evaluate bpm --source ~/music --reference metadata
```

## Reference Data Files

- `bpm_tagged.csv` - Hand-tagged BPM values for test fixtures (if available)

## Path Matching

The evaluation framework normalizes paths for matching:

1. Resolves both reference and discovered file paths to absolute paths
2. Handles WSL path translation (`/mnt/c/...` vs `C:\...`)
3. Falls back to filename-based matching if paths don't share common root

Always prefer absolute paths in reference files for reliability.
