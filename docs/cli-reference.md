# CLI Reference

## Global Options

```bash
edm [OPTIONS] COMMAND [ARGS]
```

| Option | Description |
|--------|-------------|
| `--version`, `-v` | Show version and exit |
| `--help` | Show help and exit |

## Commands

### `analyze` - Analyze Audio Files

```bash
edm analyze [OPTIONS] FILES...
```

Analyzes EDM tracks for BPM, structure, and other features.

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `FILES` | Yes | Audio files to analyze (supports globs) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--types`, `-t` | TEXT | all | Comma-separated analysis types: `bpm`, `grid`, `structure` |
| `--output`, `-o` | PATH | - | Save results to JSON file |
| `--format`, `-f` | TEXT | `table` | Output format: `table`, `json` |
| `--config`, `-c` | PATH | - | Path to configuration file |
| `--recursive`, `-r` | FLAG | - | Recursively analyze directories |
| `--ignore-metadata` | FLAG | - | Skip reading BPM from audio file metadata |
| `--structure-detector` | TEXT | `auto` | Structure detection method: `auto`, `msaf`, `energy` |
| `--log-level` | TEXT | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | PATH | - | Write logs to file (JSON format) |
| `--json-logs` | FLAG | - | Output logs in JSON format |
| `--verbose` | FLAG | - | Equivalent to `--log-level DEBUG` |
| `--quiet`, `-q` | FLAG | - | Suppress non-essential output |
| `--no-color` | FLAG | - | Disable colored output |

#### BPM Lookup Strategy

Default: metadata → computed

| Flags | Strategy |
|-------|----------|
| (none) | metadata → computed |
| `--ignore-metadata` | computed only |

#### Structure Detection Options

| Detector | Description |
|----------|-------------|
| `auto` | Use MSAF if available, fall back to energy-based (default) |
| `msaf` | Boundary detection using MSAF with energy-based labeling |
| `energy` | Rule-based detection using RMS energy analysis |

#### Examples

```bash
# Basic analysis
uv run edm analyze track.mp3

# Specific analysis types
uv run edm analyze track.mp3 --types bpm,grid

# Save results to JSON
uv run edm analyze *.mp3 --output results.json

# Recursive directory analysis
uv run edm analyze /path/to/tracks/ --recursive

# Force computation only (skip metadata)
uv run edm analyze track.mp3 --ignore-metadata

# Structure analysis only with energy detector
uv run edm analyze track.mp3 --types structure --structure-detector energy

# Debug logging
uv run edm analyze track.mp3 --log-level DEBUG --no-color
```

### `evaluate` - Accuracy Evaluation

```bash
edm evaluate [OPTIONS] COMMAND [ARGS]
```

Evaluate accuracy of analysis algorithms.

#### Subcommands

##### `evaluate bpm` - BPM Accuracy Evaluation

```bash
edm evaluate bpm [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--source` | DIRECTORY | Yes | - | Directory containing audio files |
| `--reference` | TEXT | Yes | - | Reference source: `metadata` or path to CSV/JSON |
| `--sample-size` | INTEGER | No | `100` | Number of files to sample |
| `--output` | DIRECTORY | No | `data/accuracy/bpm/` | Output directory |
| `--seed` | INTEGER | No | - | Random seed for reproducible sampling |
| `--full` | FLAG | No | - | Evaluate all files (ignore `--sample-size`) |
| `--tolerance` | FLOAT | No | `2.5` | BPM tolerance for accuracy calculation |

#### Reference Sources

| Source | Description |
|--------|-------------|
| `metadata` | Read BPM from file metadata (ID3 tags) |
| `path/to/file.csv` | CSV file with columns: `file`, `bpm` |
| `path/to/file.json` | JSON file with file-to-BPM mapping |

#### BPM Evaluation Examples

```bash
# Evaluate against CSV reference
uv run edm evaluate bpm --source ~/music --reference data/annotations/bpm_tagged.csv

# Evaluate against file metadata
uv run edm evaluate bpm --source ~/music --reference metadata

# Full evaluation with fixed seed
uv run edm evaluate bpm --source ~/music --reference metadata --full --seed 42

# Custom tolerance and output
uv run edm evaluate bpm --source ~/music --reference metadata --tolerance 1.0 --output ./results/
```

#### Output Files

Results saved to output directory:
- `<timestamp>_<commit>.json` - Full machine-readable results
- `<timestamp>_<commit>.md` - Human-readable summary
- `latest.json` - Symlink to most recent JSON
- `latest.md` - Symlink to most recent Markdown

##### `evaluate structure` - Structure Accuracy Evaluation

```bash
edm evaluate structure [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--source` | DIRECTORY | Yes | - | Directory containing audio files |
| `--reference` | PATH | Yes | - | Path to CSV file with ground truth annotations |
| `--sample-size` | INTEGER | No | `100` | Number of files to sample |
| `--output` | DIRECTORY | No | `data/accuracy/structure/` | Output directory |
| `--seed` | INTEGER | No | - | Random seed for reproducible sampling |
| `--full` | FLAG | No | - | Evaluate all files (ignore `--sample-size`) |
| `--tolerance` | FLOAT | No | `2.0` | Boundary tolerance in seconds for section matching |
| `--detector` | TEXT | No | `auto` | Structure detector: `auto`, `msaf`, `energy` |

#### Structure Reference CSV Format

CSV file with ground truth structure annotations supports both time-based and bar-based formats:

**Time-based format:**
```csv
filename,start,end,label
track1.mp3,0.0,30.5,intro
track1.mp3,30.5,91.5,buildup
track1.mp3,91.5,183.0,drop
track1.mp3,183.0,213.5,breakdown
track1.mp3,213.5,244.0,outro
```

**Bar-based format (requires BPM):**
```csv
filename,start_bar,end_bar,label,bpm
track1.mp3,1,17,intro,128
track1.mp3,17,49,buildup,128
track1.mp3,49,97,drop,128
track1.mp3,97,113,breakdown,128
track1.mp3,113,129,outro,128
```

**Note:** Bar numbering is 1-indexed (bar 1 = first bar). An 8-bar intro spans bars 1-8, and the following section starts at bar 9.

**Mixed format (bar positions preferred when available):**
```csv
filename,start,end,start_bar,end_bar,label,bpm
track1.mp3,0.0,30.5,1,17,intro,128
track1.mp3,30.5,91.5,17,49,buildup,128
```

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `filename` | TEXT | Yes | Audio filename (relative to source directory) |
| `start` | FLOAT | Conditional | Section start time in seconds (required if no `start_bar`) |
| `end` | FLOAT | Conditional | Section end time in seconds (required if no `end_bar`) |
| `start_bar` | FLOAT | Conditional | Section start bar position (requires `bpm`) |
| `end_bar` | FLOAT | Conditional | Section end bar position (requires `bpm`) |
| `label` | TEXT | Yes | Section label: `intro`, `buildup`, `drop`, `breakdown`, `outro` |
| `bpm` | FLOAT | Conditional | Track BPM (required for bar-based annotations) |

**Annotation Workflow:**
When annotating while listening to tracks, bar-based annotations are more natural:
1. Listen to track and note bar numbers shown in your music player/DJ software
2. Mark section boundaries by bar (e.g., "drop starts at bar 48")
3. Save with BPM from track metadata or detection
4. System converts bars to time automatically during evaluation

#### Structure Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| Boundary Error | Average distance from detected to true boundaries |

Metrics are calculated:
- **Overall**: Across all section types
- **Per-type**: For each section type (intro, buildup, drop, breakdown, outro)

#### Structure Evaluation Examples

```bash
# Evaluate against ground truth CSV
uv run edm evaluate structure --source ~/music --reference annotations.csv

# Full evaluation of all files
uv run edm evaluate structure --source ~/music --reference annotations.csv --full

# Custom boundary tolerance (±3 seconds)
uv run edm evaluate structure --source ~/music --reference annotations.csv --tolerance 3.0

# Use energy-based detector only
uv run edm evaluate structure --source ~/music --reference annotations.csv --detector energy
```

## Configuration

### Configuration File

Optional TOML configuration at `~/.config/edm/config.toml`:

```toml
[analysis]
detect_bpm = true
detect_structure = true
use_madmom = true  # Legacy name - controls beat_this library
use_librosa = false

[logging]
level = "WARNING"
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
