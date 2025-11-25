# Design Document: Accuracy Evaluation Framework

## Overview

This document details the technical design for the accuracy evaluation framework, which enables systematic testing and validation of EDM analysis algorithms against reference data.

## Architecture

Implemented as a library module (`edm.evaluation`) with CLI integration.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLI Layer                                         │
│  src/cli/commands/evaluate.py                                       │
│  - Click command: edm evaluate bpm/drops/key                        │
│  - Argument parsing and validation                                  │
│  - Routes to edm.evaluation API                                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            Library Module: edm.evaluation                            │
│  __init__.py - Public API: evaluate_bpm(), evaluate_drops()        │
│  common.py - Discovery, sampling, metrics, I/O                      │
│  reference.py - Reference loading (spotify, metadata, files)        │
│  evaluators/bpm.py - BPM evaluation logic                           │
└────┬──────────────────┬──────────────────┬──────────────────────────┘
     │                  │                  │
     ▼                  ▼                  ▼
┌──────────┐  ┌──────────────────┐  ┌────────────────┐
│ Common   │  │ Core Library     │  │ Results        │
│ Utilities│  │ (Reused)         │  │ Storage        │
│          │  │                  │  │                │
│ Sampling │  │ analyze_bpm()    │  │ JSON + MD      │
│ Metrics  │  │ read_metadata()  │  │ Visualization  │
│ Reference│  │ SpotifyClient    │  │ Git tracking   │
└──────────┘  └──────────────────┘  └────────────────┘
```

**Key Design Principles:**
- ✅ **Proper library module**: Code under `src/edm/evaluation/`
- ✅ **CLI integration**: `edm evaluate` command
- ✅ **Reusable**: Programmatic API for notebooks/scripts
- ✅ **Testable**: Standard pytest structure
- ✅ **Internal**: Not part of public API docs

## Core Data Structures

**Note:** These abstractions are kept simple. No need for abstract base classes unless we add 3+ analysis types. Use plain functions and dictionaries.

### Result Data Classes

Simple dictionaries for passing data between functions:

```python
# Single file result
result = {
    "file_path": str(path),
    "reference": 128.0,
    "computed": 127.8,
    "error": -0.2,
    "is_octave_error": False,
    "confidence": 0.95,  # If available from algorithm
    "success": True,
    "computation_time": 0.843
}

# Full evaluation summary
summary = {
    "metadata": {
        "analysis_type": "bpm",
        "algorithm_version": "madmom-0.16.1",
        "timestamp": "2025-11-24T14:30:52",
        "git_commit": "abc123",
        "sample_size": 100,
        "tolerance": 2.5
    },
    "summary": {
        "total_files": 100,
        "successful": 97,
        "failed": 3,
        "mean_absolute_error": 1.84,
        "root_mean_square_error": 2.91,
        "accuracy_within_tolerance": 87.6,
        "octave_errors": 5,
        "accuracy_with_octave_correction": 92.8,
        "tempo_stratified": {
            "slow_100_120": {"count": 15, "mae": 1.2, "accuracy": 93.3},
            "medium_120_135": {"count": 60, "mae": 1.5, "accuracy": 91.7},
            "fast_135_150": {"count": 20, "mae": 2.8, "accuracy": 75.0},
            "very_fast_150plus": {"count": 5, "mae": 4.1, "accuracy": 60.0}
        }
    },
    "results": [...]  # List of individual results
}
```

## Library Modules

### `src/edm/evaluation/common.py` - Core Utilities

Simple functions, no classes needed:

```python
import csv
import json
import random
from pathlib import Path
from typing import List, Dict

# File discovery
def discover_audio_files(source_path: Path) -> List[Path]:
    """Find all audio files recursively."""
    extensions = {".mp3", ".flac", ".wav", ".m4a"}
    files = []
    for ext in extensions:
        files.extend(source_path.rglob(f"*{ext}"))
    return sorted(files)

# Sampling
def sample_random(files: List[Path], size: int, seed=None) -> List[Path]:
    """Random sample with optional seed."""
    if seed:
        random.seed(seed)
    return random.sample(files, min(size, len(files)))

def sample_full(files: List[Path]) -> List[Path]:
    """Return all files."""
    return files

# Reference loading
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

def load_reference_csv(path: Path) -> Dict[Path, float]:
    """Load reference BPMs from CSV."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[Path(row["path"])] = float(row["bpm"])
    return data

def load_reference_json(path: Path) -> Dict[Path, float]:
    """Load reference BPMs from JSON."""
    with open(path) as f:
        records = json.load(f)
    return {Path(r["path"]): float(r["bpm"]) for r in records}

def load_spotify_reference(source_path: Path) -> Dict[Path, float]:
    """Load BPM data from Spotify API for discovered files."""
    from edm.external.spotify import SpotifyClient
    from edm.io.metadata import read_metadata

    logger = structlog.get_logger(__name__)
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

    logger = structlog.get_logger(__name__)
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

# Metrics
def calculate_mae(errors: List[float]) -> float:
    """Mean Absolute Error."""
    return sum(abs(e) for e in errors) / len(errors)

def calculate_rmse(errors: List[float]) -> float:
    """Root Mean Square Error."""
    return (sum(e**2 for e in errors) / len(errors)) ** 0.5

def calculate_accuracy(errors: List[float], tolerance: float) -> float:
    """Accuracy as percentage within tolerance."""
    within = sum(abs(e) <= tolerance for e in errors)
    return (within / len(errors)) * 100.0

def is_octave_error(computed: float, reference: float, tolerance: float = 2.5) -> bool:
    """Check if error is 2x or 0.5x (common BPM detection failure mode)."""
    return (abs(computed - reference * 2) <= tolerance or
            abs(computed - reference / 2) <= tolerance)

def calculate_octave_metrics(results: List[dict], tolerance: float) -> dict:
    """Calculate octave-aware accuracy metrics."""
    exact_correct = 0
    octave_correctable = 0

    for r in [x for x in results if x["success"]]:
        error = abs(r["error"])
        if error <= tolerance:
            exact_correct += 1
        elif is_octave_error(r["computed"], r["reference"], tolerance):
            octave_correctable += 1

    total = len([x for x in results if x["success"]])
    return {
        "octave_errors": octave_correctable,
        "accuracy_with_octave_correction": (exact_correct + octave_correctable) / total * 100 if total > 0 else 0
    }

def stratify_by_tempo(results: List[dict]) -> Dict[str, List[dict]]:
    """Group results by tempo ranges for stratified metrics."""
    strata = {
        "slow_100_120": [],
        "medium_120_135": [],
        "fast_135_150": [],
        "very_fast_150plus": []
    }
    for r in [x for x in results if x["success"]]:
        ref = r["reference"]
        if ref < 120:
            strata["slow_100_120"].append(r)
        elif ref < 135:
            strata["medium_120_135"].append(r)
        elif ref < 150:
            strata["fast_135_150"].append(r)
        else:
            strata["very_fast_150plus"].append(r)
    return strata

def normalize_path(path: Path, source_root: Path) -> Path:
    """Normalize path relative to source root for consistent reference matching."""
    try:
        # Resolve symlinks and make absolute
        abs_path = path.resolve()
        abs_root = source_root.resolve()
        # Convert to relative path
        return abs_path.relative_to(abs_root)
    except ValueError:
        # Path not under source_root, return as-is
        return path

# Results storage
def save_results_json(results: dict, output_path: Path):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def save_results_markdown(results: dict, output_path: Path):
    """Save summary to Markdown."""
    # Generate human-readable summary
    pass

def save_error_distribution_plot(errors: List[float], output_path: Path):
    """Save error distribution plot (optional matplotlib visualization)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib_not_available",
                      msg="Install matplotlib for visualization: pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (BPM)')
    plt.ylabel('Frequency')
    plt.title('BPM Error Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("plot_saved", path=str(output_path))
```

### `src/edm/evaluation/reference.py` - Reference Loading

Unified reference loading with auto-detection:

```python
from pathlib import Path
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)

def load_reference(
    reference_arg: str,
    analysis_type: str,
    source_path: Path,
    value_field: str = "bpm"
) -> Dict[Path, Any]:
    """Auto-detect and load reference based on argument and analysis type."""
    if reference_arg.lower() == "spotify":
        if analysis_type not in ["bpm"]:
            raise ValueError(f"Spotify reference not supported for {analysis_type}")
        return _load_spotify(source_path)

    if reference_arg.lower() == "metadata":
        if analysis_type not in ["bpm", "key"]:
            raise ValueError(f"Metadata reference not supported for {analysis_type}")
        return _load_metadata(source_path, value_field)

    ref_path = Path(reference_arg)
    if ref_path.suffix == ".csv":
        return _load_csv(ref_path, value_field)
    elif ref_path.suffix == ".json":
        return _load_json(ref_path, value_field)
    else:
        raise ValueError(f"Unknown reference format: {reference_arg}")

def _load_spotify(source_path: Path) -> Dict[Path, float]:
    """Load BPM from Spotify API with permanent caching."""
    from edm.external.spotify import SpotifyClient
    from edm.io.metadata import read_metadata

    cache_path = Path("benchmarks/results/accuracy/.cache/spotify_bpm.json")
    cache = _load_cache(cache_path)

    client = SpotifyClient()
    reference = {}
    files = discover_audio_files(source_path)

    for file_path in files:
        normalized = normalize_path(file_path, source_path)
        key = str(normalized)

        # Check cache first
        if key in cache:
            reference[file_path] = cache[key]["bpm"]
            continue

        # Fetch from Spotify
        try:
            metadata = read_metadata(file_path)
            track_info = client.search_track(
                metadata.get("artist"),
                metadata.get("title")
            )
            if track_info and track_info.get("bpm"):
                bpm = float(track_info["bpm"])
                reference[file_path] = bpm
                cache[key] = {
                    "bpm": bpm,
                    "track_id": track_info.get("id"),
                    "fetched_at": str(datetime.now().date())
                }
        except Exception as e:
            logger.warning("spotify_lookup_failed", file=str(file_path), error=str(e))

    _save_cache(cache_path, cache)
    return reference
```

### `src/edm/evaluation/evaluators/bpm.py` - BPM Evaluation

Main evaluation logic:

```python
import time
from pathlib import Path
from typing import Dict, Any
import structlog
from edm.analysis.bpm import analyze_bpm
from ..common import discover_audio_files, sample_random, sample_full
from ..common import calculate_mae, calculate_rmse, calculate_accuracy
from ..common import calculate_octave_metrics, stratify_by_tempo
from ..common import save_results_json, save_results_markdown
from ..common import save_error_distribution_plot, get_git_commit
from ..reference import load_reference

logger = structlog.get_logger(__name__)

def evaluate_bpm(
    source_path: Path,
    reference_source: str,
    sample_size: int = 100,
    full: bool = False,
    seed: int | None = None,
    tolerance: float = 2.5,
    output_dir: Path | None = None
) -> Dict[str, Any]:
    """Run BPM evaluation and return results."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/accuracy/bpm")

    # 1. Discover files
    files = discover_audio_files(source_path)
    logger.info("discovered_files", count=len(files))

    # 2. Sample
    if full:
        sampled = sample_full(files)
    else:
        sampled = sample_random(files, sample_size, seed)
    logger.info("sampled_files", count=len(sampled))

    # 3. Load reference
    reference = load_reference(
        reference_source,
        analysis_type="bpm",
        source_path=source_path
    )

    # 4. Evaluate each file
    results = []
    for i, file_path in enumerate(sampled, 1):
        ref_value = reference.get(file_path)
        if ref_value is None:
            logger.warning("no_reference", file=str(file_path))
            continue

        try:
            start = time.time()
            computed = analyze_bpm(file_path, force_compute=True, offline=True)
            elapsed = time.time() - start

            error = computed - ref_value
            results.append({
                "file_path": str(file_path),
                "reference": ref_value,
                "computed": computed,
                "error": error,
                "success": True,
                "computation_time": elapsed
            })
            logger.info("evaluated", file=file_path.name, ref=ref_value,
                       computed=computed, error=error)
        except Exception as e:
            logger.error("evaluation_failed", file=str(file_path), error=str(e))
            results.append({
                "file_path": str(file_path),
                "reference": ref_value,
                "computed": None,
                "error": None,
                "success": False,
                "error_message": str(e)
            })

    # 5. Calculate metrics
    errors = [r["error"] for r in results if r["success"]]
    octave_metrics = calculate_octave_metrics(results, tolerance)
    tempo_strata = stratify_by_tempo(results)

    summary = {
        "total_files": len(sampled),
        "successful": len(errors),
        "failed": len(results) - len(errors),
        "mean_absolute_error": calculate_mae(errors),
        "root_mean_square_error": calculate_rmse(errors),
        "accuracy_within_tolerance": calculate_accuracy(errors, tolerance),
        **octave_metrics,
        "tempo_stratified": {
            name: {
                "count": len(strata),
                "mae": calculate_mae([r["error"] for r in strata]),
                "accuracy": calculate_accuracy([r["error"] for r in strata], tolerance)
            }
            for name, strata in tempo_strata.items() if len(strata) > 0
        }
    }

    # 6. Save results
    output = {
        "metadata": {
            "analysis_type": "bpm",
            "algorithm_version": "madmom-0.16.1",  # Get from edm.analysis.bpm
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": get_git_commit(),
            "sample_size": len(sampled),
            "sampling_strategy": "full" if full else "random",
            "sampling_seed": seed,
            "tolerance": tolerance
        },
        "summary": summary,
        "results": results
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.strftime('%Y-%m-%d')}_bpm_eval_commit-{get_git_commit()[:7]}.json"
    save_results_json(output, output_path)
    save_results_markdown(output, output_path.with_suffix('.md'))

    # Create symlinks
    (output_dir / "latest.json").unlink(missing_ok=True)
    (output_dir / "latest.json").symlink_to(output_path.name)
    (output_dir / "latest.md").unlink(missing_ok=True)
    (output_dir / "latest.md").symlink_to(output_path.with_suffix('.md').name)

    # Save visualization (optional)
    if errors:
        save_error_distribution_plot(errors, output_path.with_suffix('.png'))

    logger.info("evaluation_complete", path=str(output_path))
    return output
```

### `src/cli/commands/evaluate.py` - CLI Command

Click-based CLI integration:

```python
import click
from pathlib import Path
from edm.evaluation import evaluate_bpm

@click.group()
def evaluate():
    """Evaluate analysis accuracy against reference data."""
    pass

@evaluate.command()
@click.option("--source", required=True, type=click.Path(exists=True),
              help="Path to music directory")
@click.option("--reference", required=True,
              help="Reference source: 'spotify', 'metadata', or path to CSV/JSON file")
@click.option("--sample-size", default=100, type=int,
              help="Number of files to sample (default: 100)")
@click.option("--full", is_flag=True,
              help="Use all files instead of sampling")
@click.option("--seed", type=int,
              help="Random seed for reproducible sampling")
@click.option("--tolerance", default=2.5, type=float,
              help="BPM tolerance for accuracy metric (default: 2.5)")
@click.option("--output", type=click.Path(),
              help="Output directory (default: benchmarks/results/accuracy/bpm/)")
def bpm(source, reference, sample_size, full, seed, tolerance, output):
    """Evaluate BPM detection accuracy."""
    output_dir = Path(output) if output else None

    results = evaluate_bpm(
        source_path=Path(source),
        reference_source=reference,
        sample_size=sample_size,
        full=full,
        seed=seed,
        tolerance=tolerance,
        output_dir=output_dir
    )

    summary = results["summary"]
    click.echo(f"\n{'='*50}")
    click.echo("Evaluation Complete!")
    click.echo(f"{'='*50}")
    click.echo(f"MAE: {summary['mean_absolute_error']:.2f} BPM")
    click.echo(f"RMSE: {summary['root_mean_square_error']:.2f} BPM")
    click.echo(f"Accuracy (±{tolerance} BPM): {summary['accuracy_within_tolerance']:.1f}%")
    click.echo(f"Octave-aware accuracy: {summary['accuracy_with_octave_correction']:.1f}%")
```

Then register in `src/cli/main.py`:

```python
from cli.commands.evaluate import evaluate

# Add to main CLI group
cli.add_command(evaluate)
```

## Output Format

**JSON** (`benchmarks/results/accuracy/bpm/bpm-eval-2025-11-24-143052.json`):
```json
{
  "metadata": {
    "analysis_type": "bpm",
    "timestamp": "2025-11-24T14:30:52",
    "sample_size": 100,
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

## Design Notes

### Error Handling
- **Missing reference**: Log warning, skip file
- **Analysis failure**: Log error, record as failed, continue
- **Invalid paths**: Validate early, fail fast

### Logging
Use structlog for structured logging throughout:
```python
logger.info("discovered_files", count=1523)
logger.warning("no_reference", file="track.mp3")
logger.error("evaluation_failed", file="track.mp3", error="...")
```

### Extension
To add new analysis types (drops, key):
1. Create `benchmarks/accuracy/drops.py` with `evaluate_drops(args)`
2. Add subparser in `benchmarks/accuracy/evaluate.py`
3. Reuse functions from `common.py`

### Core Library Reuse
- ✅ `edm.analysis.bpm.analyze_bpm()` - BPM detection
- ✅ `edm.io.metadata.read_metadata()` - File metadata
- ✅ `edm.external.spotify.SpotifyClient` - Spotify API (optional)
- ✅ `structlog` - Structured logging

### Performance
- Scripts are synchronous - simplicity over speed
- For 100 files @ 0.8s each = ~80s total
- Parallel processing can be added later if needed

### Future Enhancements
- Markdown output alongside JSON
- Git commit tracking in metadata
- Symlinks to `latest.json`
- Progress bars (tqdm)
- Visualization (matplotlib)
