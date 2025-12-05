# Data Management Specification

## Overview

This specification defines the data management system for EDM track structure annotations, including the annotation schema, quality tiers, versioning with DVC, and CLI tools.

## Annotation Schema

### File Format

Annotations are stored as YAML files in `data/annotations/` with the following structure:

```yaml
metadata:
  tier: 1  # Quality tier (1=gold, 2=silver, 3=bronze)
  confidence: 0.95  # Overall confidence [0-1]
  source: "manual"  # Source: manual, rekordbox, auto, algorithm_name
  created: "2025-12-05T00:00:00Z"  # ISO 8601 timestamp
  modified: "2025-12-05T00:00:00Z"  # ISO 8601 timestamp
  verified_by: "username"  # Optional: who verified this
  notes: "Optional notes"  # Optional: freeform notes
  flags: []  # List of flags: needs_review, high_confidence, etc.
  validation:  # Auto-populated validation results
    beat_grid_valid: true
    cue_points_snapped: true
    min_section_length: true

audio:
  file: /path/to/audio.flac  # Absolute path to audio file
  duration: 300.0  # Duration in seconds (float)
  bpm: 128.0  # Beats per minute
  downbeat: 0.453  # First downbeat time in seconds
  time_signature: [4, 4]  # [numerator, denominator]
  key: "Am"  # Optional: musical key

structure:
  - bar: 1  # 1-indexed bar number
    label: "intro"  # Section label
    time: 0.0  # Time in seconds
    confidence: 0.95  # Section-specific confidence
  - bar: 9
    label: "breakdown"
    time: 14.5
    confidence: 0.9
  # ... more sections

energy:  # Optional: energy analysis
  overall: 0.7  # Overall track energy [0-1]
  by_section:
    bass: [0.1, 0.3, 0.9, ...]  # Per-section energy
    mid: [0.2, 0.4, 0.8, ...]
    high: [0.1, 0.3, 0.7, ...]
  at_boundaries: [0.1, 0.5, 0.9, ...]  # Energy at section boundaries
```

### Required Fields

- `metadata.tier`: Must be 1, 2, or 3
- `metadata.confidence`: Must be in [0, 1]
- `metadata.source`: String identifying source
- `metadata.created`: ISO 8601 timestamp
- `metadata.modified`: ISO 8601 timestamp
- `audio.file`: Path to audio file
- `audio.duration`: Duration in seconds (float)
- `audio.bpm`: BPM (float > 0)
- `audio.downbeat`: First downbeat time
- `audio.time_signature`: Array of 2 integers
- `structure`: List of at least one section

### Optional Fields

- `metadata.verified_by`: Username string
- `metadata.notes`: Freeform text
- `metadata.flags`: Array of flag strings
- `audio.key`: Musical key string
- `energy`: Full energy analysis object

### Section Labels

Valid section labels (EDM terminology):
- `intro`: Opening section (8-32 bars)
- `buildup`: Tension section (8-16 bars)
- `drop`: Peak energy section / chorus (16-32 bars)
- `breakdown`: Contrast section / verse (8-16 bars)
- `outro`: Closing section (16-32 bars)

## Quality Tier System

### Tier 1: Gold Standard
- **Source**: Manually created and verified by expert
- **Confidence**: ≥ 0.9
- **Use case**: Validation sets, model evaluation, final metrics
- **Requirements**:
  - All sections verified
  - Beat grid accurate
  - Section boundaries on downbeats
  - Minimum section length met (4 bars)

### Tier 2: Silver Standard
- **Source**: Imported from DJ software (Rekordbox) or semi-automated
- **Confidence**: 0.7 - 0.9
- **Use case**: Training data with sample weighting
- **Requirements**:
  - Plausible section boundaries
  - BPM validated
  - May have minor alignment issues

### Tier 3: Bronze Standard
- **Source**: Fully automated, unchecked
- **Confidence**: < 0.7
- **Use case**: Exploratory analysis, bulk processing
- **Requirements**:
  - Schema-compliant only
  - No quality guarantees

## Data Versioning with DVC

### Tracked Data

DVC tracks the following directories:
- `data/annotations/`: All annotation YAML files
- `data/models/`: Trained model checkpoints
- `data/generated/`: Generated/processed data

### Workflow

1. **Make changes**: Edit annotations or train models
2. **Add to DVC**: `dvc add data/annotations`
3. **Commit**:
   ```bash
   git add data/annotations.dvc
   git commit -m "update annotations: add 10 tier-1 tracks"
   ```
4. **Push data**: `dvc push` (to remote storage)
5. **Push git**: `git push`

### Pipeline Integration

The `dvc.yaml` defines a training pipeline:

```yaml
stages:
  train:
    cmd: uv run edm train data/annotations --audio-dir ~/music
    deps:
      - data/annotations
      - src/edm/training
    outs:
      - outputs/training/checkpoints
    metrics:
      - outputs/training/metrics.json
```

Run pipeline: `dvc repro train`

This automatically:
- Reruns training when annotations change
- Tracks model checkpoints
- Links data version to model version

## CLI Commands

### edm data stats

Display annotation statistics:

```bash
edm data stats [PATH]
```

Output:
- Total annotations
- Count by tier
- Average confidence by tier
- Flagged tracks
- Validation issues

### edm data validate

Validate all annotations against schema and heuristics:

```bash
edm data validate [PATH] [--strict]
```

Checks:
- Schema compliance
- Bar numbers in ascending order
- Times in ascending order
- Minimum section length (4 bars recommended)
- Beat grid snapping
- BPM consistency

### edm data export

Export annotations to various formats:

```bash
edm data export [PATH] --format {json,pytorch,weights} --output OUTPUT
```

Formats:
- `json`: Single JSON file with all annotations
- `pytorch`: PyTorch-compatible dataset directory
- `weights`: CSV of confidence weights for training

### edm data tier

Update tier for annotations:

```bash
edm data tier --set TIER FILE [FILE ...]
```

Example:
```bash
edm data tier --set 1 track1.yaml track2.yaml
```

### edm data flag

Add/remove flags:

```bash
edm data flag --add FLAG FILE [FILE ...]
edm data flag --remove FLAG FILE [FILE ...]
```

Common flags:
- `needs_review`: Annotation needs manual review
- `high_confidence`: Exceptionally high quality
- `questionable_bpm`: BPM may be incorrect
- `boundary_issues`: Section boundaries may be misaligned

## Validation Rules

### Schema Validation

1. All required fields present
2. Types correct (int, float, string, array)
3. Confidence in [0, 1]
4. Tier in {1, 2, 3}

### Heuristic Validation

1. **Bar numbers**: Strictly ascending
2. **Times**: Strictly ascending
3. **Section length**: ≥ 4 bars (warning if shorter)
4. **Beat snapping**: Section times within tolerance of beat grid
5. **BPM range**: 60-200 BPM (typical EDM range)

## Integration with Training

### PyTorch Dataset

The `EDMDataset` class supports:
- **Tier filtering**: `tier_filter=1` for gold-standard validation sets
- **Confidence filtering**: `min_confidence=0.8` to exclude low-quality data
- **Sample weighting**: Uses `metadata.confidence` for weighted loss

Example:
```python
from edm.training import EDMDataset

# Tier-1 validation set
val_dataset = EDMDataset(
    annotation_dir="data/annotations",
    tier_filter=1,
    min_confidence=0.9
)

# All tiers for training with confidence weighting
train_dataset = EDMDataset(
    annotation_dir="data/annotations",
    tier_filter=None,  # All tiers
    min_confidence=0.5  # Exclude very low confidence
)
```

### Loss Weighting

Confidence scores are passed to the training loop and can be used for sample weighting:

```python
loss = criterion(outputs, targets) * confidence_weights
```

This down-weights low-confidence samples during training.

## Migration from Old Format

The old format stored annotations as simple arrays:

```yaml
annotations:
  - [1, "intro", 0.0]
  - [9, "breakdown", 14.5]
```

This is **explicitly rejected** by the new schema validator to prevent mixing formats.

To migrate:
1. Commit old annotations: `git commit -m "archive: old format before DATAMGMT"`
2. Delete old files: `rm data/annotations/*.yaml`
3. Re-import using new converters with metadata

## Best Practices

### Creating Annotations

1. **Import from DJ software**: Use `edm import rekordbox` for existing cue points
2. **Set appropriate tier**: Default to Tier 2 for imports, Tier 1 after manual verification
3. **Add notes**: Document any uncertainty or special cases
4. **Flag for review**: Use `needs_review` flag liberally

### Maintaining Quality

1. **Validate regularly**: Run `edm data validate` before committing
2. **Review flagged tracks**: Check tracks with `needs_review` flag
3. **Version with DVC**: Always `dvc add` after changes
4. **Track confidence**: Lower confidence for uncertain sections

### Training Strategies

1. **Tier-1 validation**: Always use Tier 1 for final metrics
2. **Mixed training**: Train on all tiers with confidence weighting
3. **Confidence threshold**: Filter out confidence < 0.5 for training
4. **Separate test set**: Reserve Tier-1 tracks for testing only

## References

- Schema implementation: `src/edm/data/schema.py`
- Validation: `src/edm/data/validation.py`
- CLI commands: `src/cli/commands/data.py`
- Dataset: `src/edm/training/dataset.py`
- DVC configuration: `dvc.yaml`, `params.yaml`
