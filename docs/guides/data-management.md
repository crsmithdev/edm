# Data Management

Guide to managing annotations, datasets, and data workflows in the EDM project.

## Table of Contents

- [Overview](#overview)
- [Annotation Formats](#annotation-formats)
- [Import & Export](#import--export)
- [Validation](#validation)
- [Dataset Management](#dataset-management)
- [Best Practices](#best-practices)

---

## Overview

EDM uses custom YAML annotations for training and reference data. The system supports:

- **Import**: Rekordbox XML → EDM YAML
- **Export**: EDM YAML → JAMS format
- **Validation**: Schema and consistency checks
- **Versioning**: DVC tracking for datasets

### Directory Structure

```
data/
├── annotations/              # Annotation files
│   ├── reference/            # Hand-annotated ground truth
│   │   ├── track1.yaml
│   │   ├── track2.yaml
│   │   └── ...
│   └── generated/            # Auto-generated annotations
│       ├── track1.yaml
│       └── ...
├── jams/                     # Exported JAMS format
└── accuracy/                 # Evaluation results
    ├── bpm/
    └── structure/
```

---

## Annotation Formats

### EDM YAML Format

**Complete example**:

```yaml
audio:
  file: music/track.mp3
  duration: 360.5
  bpm: 128.0
  downbeat: 0.5
  time_signature: [4, 4]

structure:
  sections:
    - label: intro
      start_time: 0.0
      end_time: 32.0
      start_bar: 1
      end_bar: 17
      confidence: 1.0

    - label: buildup
      start_time: 32.0
      end_time: 48.0
      start_bar: 17
      end_bar: 25
      confidence: 1.0

    - label: drop
      start_time: 48.0
      end_time: 96.0
      start_bar: 25
      end_bar: 49
      confidence: 1.0

metadata:
  annotator: human
  created: 2025-01-15
  version: 1.0
```

**Minimal example** (for training):

```yaml
audio:
  file: music/track.mp3
  bpm: 128.0
  downbeat: 0.5

structure:
  sections:
    - label: intro
      start_time: 0.0
      end_time: 32.0

    - label: drop
      start_time: 32.0
      end_time: 96.0
```

### Time-Based vs Bar-Based

**Time-based** (absolute timestamps):
```yaml
- label: drop
  start_time: 48.0
  end_time: 96.0
```

**Bar-based** (musical bars):
```yaml
- label: drop
  start_bar: 25
  end_bar: 49
  # Requires BPM and downbeat for conversion
```

Bar formula: `time = downbeat + (bar - 1) × (60 / bpm) × 4`

### Section Labels

Standard EDM structure labels:

| Label | Description | Typical Length |
|-------|-------------|----------------|
| `intro` | Low energy opening | 8-32 bars |
| `buildup` | Rising energy, tension | 8-16 bars |
| `drop` | Peak energy, full arrangement | 16-32 bars |
| `breakdown` | Mid energy, reduced elements | 8-16 bars |
| `outro` | Falling energy, fadeout | 16-32 bars |

For complete terminology, see [Terminology Guide](../reference/terminology.md).

### CSV Format (for evaluation)

**Time-based**:
```csv
filename,start,end,label
track1.mp3,0.0,30.5,intro
track1.mp3,30.5,78.2,drop
track1.mp3,78.2,125.0,breakdown
```

**Bar-based**:
```csv
filename,start_bar,end_bar,label,bpm
track1.mp3,1,17,intro,128.0
track1.mp3,17,49,drop,128.0
track1.mp3,49,81,breakdown,128.0
```

### JAMS Format

JAMS (JSON Annotated Music Specification) for interoperability:

```json
{
  "file_metadata": {
    "title": "track",
    "duration": 360.5
  },
  "annotations": [
    {
      "namespace": "segment_open",
      "data": [
        {
          "time": 0.0,
          "duration": 32.0,
          "value": "intro",
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

---

## Import & Export

### Import from Rekordbox

Rekordbox XML contains BPM, downbeats, and phrase markers:

```bash
# Import Rekordbox library
uv run edm data import-rekordbox ~/music/rekordbox.xml

# Output: Creates YAML files in data/annotations/
# - BPM from track metadata
# - Downbeat from first beat marker
# - Sections from phrase markers
```

**Rekordbox phrase mapping**:

| Rekordbox | EDM Label |
|-----------|-----------|
| Intro | intro |
| Up | buildup |
| Chorus | drop |
| Verse | breakdown |
| Outro | outro |

### Export to JAMS

Export EDM annotations to JAMS format for research/comparison:

```bash
# Export all annotations
uv run edm data export-jams data/annotations --output jams/

# Export specific files
uv run edm data export-jams data/annotations/track1.yaml --output jams/track1.jams
```

JAMS is used by:
- MIREX (Music Information Retrieval Evaluation eXchange)
- mir_eval evaluation library
- Research publications

---

## Validation

### Validate Annotations

Check annotation files for errors:

```bash
# Validate all annotations
uv run edm data validate data/annotations

# Validate specific file
uv run edm data validate data/annotations/track1.yaml

# Verbose output
uv run edm data validate data/annotations --verbose
```

**Checks performed**:
- ✅ YAML syntax
- ✅ Required fields present
- ✅ Valid label values
- ✅ Time ranges non-overlapping
- ✅ Sections in chronological order
- ✅ Bar calculations (if provided)
- ✅ Audio file exists (if path valid)

**Example output**:

```
Validating data/annotations/reference/

✓ track1.yaml (5 sections)
✗ track2.yaml
  - Missing required field: audio.bpm
  - Invalid label: "break" (should be "breakdown")
  - Overlapping sections: buildup (32.0-48.0) and drop (45.0-96.0)
✓ track3.yaml (6 sections)

Summary: 2/3 valid, 1 error
```

### Validation Script

Check annotations programmatically:

```python
import yaml
from pathlib import Path

annotation_dir = Path('data/annotations')

for file in annotation_dir.glob('*.yaml'):
    ann = yaml.safe_load(file.read_text())

    # Check audio file exists
    audio_path = Path(ann['audio']['file'])
    if not audio_path.exists():
        print(f'{file.name}: Missing audio file: {audio_path}')

    # Check required fields
    if 'bpm' not in ann['audio']:
        print(f'{file.name}: Missing BPM')

    # Check sections
    sections = ann['structure']['sections']
    for i, section in enumerate(sections):
        if section['label'] not in ['intro', 'buildup', 'drop', 'breakdown', 'outro']:
            print(f'{file.name}: Invalid label: {section["label"]}')

        # Check chronological order
        if i > 0 and section['start_time'] < sections[i-1]['end_time']:
            print(f'{file.name}: Sections out of order')
```

---

## Dataset Management

### Audio File Management

**Check audio files**:

```bash
# Find all audio files
find ~/music -name "*.flac" -o -name "*.mp3" | wc -l

# Check audio file metadata
uv run python -c "
import mutagen
from pathlib import Path
audio = mutagen.File(Path('~/music/track.flac').expanduser())
print(f'Duration: {audio.info.length}s')
print(f'Sample rate: {audio.info.sample_rate}Hz')
print(f'Bitrate: {audio.info.bitrate / 1000}kbps')
"
```

**Supported formats**:
- MP3 (.mp3)
- FLAC (.flac)
- WAV (.wav)
- M4A (.m4a)

### Verify Training Dataset

```bash
# Count annotations
ls data/annotations/*.yaml | wc -l

# Validate all annotations
uv run edm data validate data/annotations

# Check audio paths in annotations
python -c "
import yaml
from pathlib import Path
for f in Path('data/annotations').glob('*.yaml'):
    ann = yaml.safe_load(f.read_text())
    audio = Path(ann['audio']['file'])
    if not audio.exists():
        print(f'Missing: {audio}')
"

# Test dataset loading
uv run python -c "
from edm.training.dataset import EDMDataset
from pathlib import Path
dataset = EDMDataset(
    annotation_dir=Path('data/annotations'),
    audio_dir=Path('~/music').expanduser(),
)
print(f'Dataset size: {len(dataset)}')
print(f'First sample keys: {list(dataset[0].keys())}')
"
```

### Environment Configuration

Set audio and annotation directories:

```bash
export EDM_AUDIO_DIR=/path/to/music
export EDM_ANNOTATION_DIR=/path/to/annotations

# Verify
echo $EDM_AUDIO_DIR
ls $EDM_AUDIO_DIR | head
```

### DVC Tracking

Version control for datasets:

```bash
# Track annotation dataset
dvc add data/annotations
git add data/annotations.dvc .gitignore
git commit -m "track annotation dataset v1.0"

# Push to remote
dvc push data/annotations.dvc

# Teammate retrieves
dvc pull data/annotations.dvc
```

For complete DVC workflow, see [Model Management Guide](model-management.md#dvc-data--model-versioning).

---

## Best Practices

### Annotation Guidelines

1. **Use musical bars for boundaries**
   - More natural than timestamps
   - Aligns with DJ/production tools
   - Easier to verify by ear

2. **Label sections accurately**
   - `drop` = peak energy (chorus equivalent)
   - `breakdown` = reduced energy (verse equivalent)
   - `buildup` = rising tension (pre-chorus)
   - See [terminology guide](../reference/terminology.md) for details

3. **Set downbeat precisely**
   - Use first beat of bar 1
   - Critical for bar calculations
   - Verify with click track if needed

4. **Include metadata**
   - Annotator name
   - Date created
   - Version number
   - Notes on any ambiguities

### Dataset Organization

**Reference vs Generated**:

```
data/annotations/
├── reference/          # Hand-annotated (high quality)
│   └── track1.yaml
└── generated/          # Auto-generated (needs review)
    └── track2.yaml
```

**Split by quality**:
- `reference/` - Training data
- `generated/` - Evaluation/comparison only

**Naming convention**:
- Use original filename: `track_name.yaml`
- Or artist-title: `artist_-_title.yaml`
- Consistent casing: lowercase recommended

### Quality Control

**Before training**:

```bash
# 1. Validate all annotations
uv run edm data validate data/annotations/reference

# 2. Check audio files exist
python -c "
import yaml
from pathlib import Path
for f in Path('data/annotations/reference').glob('*.yaml'):
    ann = yaml.safe_load(f.read_text())
    audio = Path(ann['audio']['file'])
    assert audio.exists(), f'Missing: {audio}'
print('All audio files found')
"

# 3. Verify dataset size
ls data/annotations/reference/*.yaml | wc -l  # Should be 10+ files
```

**Minimum dataset size**:
- Quick test: 5-10 tracks
- Production training: 50+ tracks
- Research quality: 100+ tracks

### Version Control

**Track dataset versions**:

```bash
# Create dataset version
dvc add data/annotations/reference
git add data/annotations/reference.dvc
git commit -m "dataset v1.2: add 20 tracks, fix 5 annotations"
git tag dataset-v1.2

# Push
dvc push
git push --tags
```

**Changelog** (in commit messages):

```
dataset v1.2: add 20 tracks, fix 5 annotations

Added:
- 20 new techno tracks (128-140 BPM)

Fixed:
- track1.yaml: corrected downbeat
- track2.yaml: relabeled breakdown as buildup
- track3.yaml: extended outro section

Statistics:
- Total tracks: 75
- Avg duration: 6.5 min
- BPM range: 120-150
```

---

## See Also

- **[Training Guide](training.md)** - Using annotations for training
- **[Model Management](model-management.md)** - Versioning datasets with DVC
- **[CLI Reference](../reference/cli.md)** - Data command reference
- **[Terminology](../reference/terminology.md)** - EDM structure terminology
- **[Annotator Guide](annotator.md)** - Web-based annotation tool
