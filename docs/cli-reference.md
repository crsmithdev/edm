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
| `--offline` | FLAG | - | Skip network lookups (Spotify API) |
| `--ignore-metadata` | FLAG | - | Skip reading metadata from audio files |
| `--log-level` | TEXT | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | PATH | - | Write logs to file (JSON format) |
| `--json-logs` | FLAG | - | Output logs in JSON format |
| `--verbose` | FLAG | - | Equivalent to `--log-level DEBUG` |
| `--quiet`, `-q` | FLAG | - | Suppress non-essential output |
| `--no-color` | FLAG | - | Disable colored output |

#### BPM Lookup Strategy

Default: metadata → Spotify → computed

| Flags | Strategy |
|-------|----------|
| (none) | metadata → spotify → computed |
| `--offline` | metadata → computed |
| `--ignore-metadata` | spotify → computed |
| `--offline --ignore-metadata` | computed only |

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

# Skip Spotify API (use metadata or compute)
uv run edm analyze track.mp3 --offline

# Force computation only
uv run edm analyze track.mp3 --offline --ignore-metadata

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
| `--reference` | TEXT | Yes | - | Reference source: `spotify`, `metadata`, or path to CSV/JSON |
| `--sample-size` | INTEGER | No | `100` | Number of files to sample |
| `--output` | DIRECTORY | No | `benchmarks/results/accuracy/bpm/` | Output directory |
| `--seed` | INTEGER | No | - | Random seed for reproducible sampling |
| `--full` | FLAG | No | - | Evaluate all files (ignore `--sample-size`) |
| `--tolerance` | FLOAT | No | `2.5` | BPM tolerance for accuracy calculation |

#### Reference Sources

| Source | Description |
|--------|-------------|
| `spotify` | Query Spotify API (requires credentials) |
| `metadata` | Read BPM from file metadata (ID3 tags) |
| `path/to/file.csv` | CSV file with columns: `file`, `bpm` |
| `path/to/file.json` | JSON file with file-to-BPM mapping |

#### Examples

```bash
# Evaluate against CSV reference
uv run edm evaluate bpm --source ~/music --reference tests/fixtures/reference/bpm_tagged.csv

# Evaluate against Spotify API
uv run edm evaluate bpm --source ~/music --reference spotify

# Evaluate against file metadata
uv run edm evaluate bpm --source ~/music --reference metadata

# Full evaluation with fixed seed
uv run edm evaluate bpm --source ~/music --reference metadata --full --seed 42

# Custom tolerance and output
uv run edm evaluate bpm --source ~/music --reference spotify --tolerance 1.0 --output ./results/
```

#### Output Files

Results saved to output directory:
- `<timestamp>_<commit>.json` - Full machine-readable results
- `<timestamp>_<commit>.md` - Human-readable summary
- `latest.json` - Symlink to most recent JSON
- `latest.md` - Symlink to most recent Markdown

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SPOTIFY_CLIENT_ID` | Spotify API client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify API client secret |

Set in `.env` file in project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

### Configuration File

Optional TOML configuration at `~/.config/edm/config.toml`:

```toml
[analysis]
detect_bpm = true
detect_structure = true
use_madmom = true
use_librosa = false

[external_services]
enable_beatport = true
enable_tunebat = true
cache_ttl = 3600

[logging]
level = "INFO"
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
