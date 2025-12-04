---
status: draft
created: 2025-12-04
---

# [DATAMGMT] Data Management Overhaul

## Why

### Current Pain Points
The current YAML-based annotation system has significant limitations:
- **No versioning:** Changes to annotations not tracked beyond git commits
- **No UI:** Manual YAML editing is error-prone and tedious
- **No metadata:** Can't track confidence scores, tiers (verified/noisy), validation status
- **No lineage:** Can't trace which annotations came from which Rekordbox export or ML model version
- **Generated data chaos:** `data/generated/` outputs not tracked, can't reproduce or compare versions
- **No collaboration tools:** Hard for multiple people to review/correct annotations
- **No quality metrics:** Can't see dataset statistics (# verified, average confidence, etc.)

### ML Training Requirements
The MLPIVOT proposal requires robust data management:
- Track confidence scores per annotation
- Manage tiers: Tier 1 (verified), Tier 2 (auto-cleaned), Tier 3 (auto-generated)
- Version control training datasets as they evolve (5 new tracks/day)
- Compare model performance across dataset versions
- Flag annotations needing manual review
- Export annotations for different ML frameworks

### Research from GitHub

Investigated existing solutions:
1. **DVC (Data Version Control)** - Git-like versioning for datasets ([dvc.org](https://dvc.org/))
2. **Label Studio** - Full annotation platform with UI ([labelstud.io](https://labelstud.io/))
3. **Cleanlab** - Noisy label detection ([cleanlab on PyPI](https://pypi.org/project/cleanlab/))
4. **Audino** - Audio annotation tool ([midas-research/audino](https://github.com/midas-research/audino))

## What

### Recommended Solution: Hybrid Approach

**Don't build custom:** Use existing tools where possible.

**Architecture:**
1. **DVC** for dataset versioning (primary)
2. **Label Studio** for annotation UI (optional, Phase 2)
3. **Cleanlab** for error detection (already planned in MLPIVOT)
4. **Enhanced YAML** with metadata fields

### Phase 1: DVC Integration (Immediate)

**Add to project:**
- `dvc.yaml` - Pipeline definitions
- `.dvc/` directory - DVC configuration
- `.dvc/config` - Remote storage setup
- `data.dvc` - Track data/ directory

**DVC will track:**
- `data/annotations/` - All annotation YAML files
- `data/generated/` - Model outputs, analysis results
- `data/models/` - Trained model checkpoints (future)
- `data/features/` - Cached feature extractions (future)

**Benefits:**
- Git-like commands: `dvc add`, `dvc push`, `dvc pull`, `dvc diff`
- Remote storage: AWS S3, Google Cloud, local NAS, or shared folder
- Dataset versioning: Tag datasets by version (v1.0, v1.1, etc.)
- Reproducibility: Track which data version produced which model
- Large file support: Don't bloat git repo

**Example workflow:**
```bash
# Track new annotations
dvc add data/annotations/
git add data/annotations.dvc .gitignore
git commit -m "add 35 new verified tracks"

# Push to remote storage
dvc push

# Pull specific version
git checkout v1.0
dvc pull
```

### Phase 2: New Annotation Format (Immediate)

**Complete redesign** of YAML structure with metadata as first-class:

```yaml
# Metadata (required)
metadata:
  tier: 1  # 1=verified, 2=auto-cleaned, 3=auto-generated
  confidence: 0.95  # Overall track confidence [0-1]
  source: rekordbox  # rekordbox | msaf | ml_model_v1 | manual
  created: 2025-12-04T10:30:00Z
  modified: 2025-12-04T11:45:00Z
  verified_by: user  # user | null
  notes: "Corrected intro boundary"
  flags: [needs_review]  # Optional flags
  validation:
    beat_grid_valid: true
    cue_points_snapped: true
    min_section_length: true

# Audio metadata (required)
audio:
  file: /path/to/track.flac
  duration: 242.20  # seconds (not M:SS.mm)
  bpm: 128.0
  downbeat: 0.02
  time_signature: [4, 4]  # [numerator, denominator]
  key: 8A  # Camelot key notation

# Structure annotations (required)
structure:
  - bar: 1
    label: intro
    time: 0.004
    confidence: 0.90
  - bar: 25
    label: buildup
    time: 45.003
    confidence: 0.85

# Energy data (optional, computed)
energy:
  overall: 0.72
  by_section:
    - section: 0  # Index into structure
      bass: 0.45
      mid: 0.68
      high: 0.82
  at_boundaries:
    - boundary: 0  # Index into structure
      delta: 0.25  # Energy change at this boundary
```

**Breaking change:** Old YAML format is invalid. Must regenerate all annotations.

### Phase 3: Label Studio Integration (Optional, Week 3-4)

**Setup:**
- Install Label Studio: `pip install label-studio`
- Create project: `label-studio init edm-annotations`
- Import annotations via Python SDK
- Configure audio segment annotation template

**Features:**
- Web UI for reviewing/editing annotations at http://localhost:8080
- Audio waveform visualization
- Collaborative review (multiple annotators)
- Keyboard shortcuts for fast labeling
- Export to YAML, JSON, CSV

**Use cases:**
- Manual verification of Tier 2/3 annotations
- Active learning: Review high-disagreement tracks
- Bulk editing: Apply corrections to multiple tracks
- QA workflow: Second annotator reviews first annotator's work

**Python SDK integration:**
```python
from label_studio_sdk import Client

ls = Client(url='http://localhost:8080', api_key='...')
project = ls.get_project(id=1)

# Import annotation
project.import_tasks([{
    'data': {'audio': '/path/to/track.flac'},
    'annotations': [...],
    'predictions': [...]  # From ML model
}])

# Export verified annotations
tasks = project.get_tasks()
for task in tasks:
    if task['is_labeled']:
        save_to_yaml(task)
```

### Phase 4: Data Management CLI (Week 2)

**New command:** `edm data`

Subcommands:
- `edm data stats` - Dataset statistics (# tracks, tiers, average confidence)
- `edm data validate` - Validate all annotations against schema
- `edm data export` - Export to ML format (PyTorch, TensorFlow, JSON)
- `edm data diff v1.0 v1.1` - Compare dataset versions
- `edm data flag --needs-review` - Flag tracks for manual review
- `edm data tier --set 1 track1.yaml track2.yaml` - Update tiers

**Example:**
```bash
# Get dataset overview
$ edm data stats
Dataset Statistics:
  Total tracks: 350
  Tier 1 (verified): 55 (15.7%)
  Tier 2 (auto-cleaned): 280 (80.0%)
  Tier 3 (auto-generated): 15 (4.3%)
  Average confidence: 0.82
  Needs review: 12 tracks

# Validate all annotations
$ edm data validate
✓ 338 annotations valid
✗ 12 annotations need review:
  - track1.yaml: cue point not on beat
  - track2.yaml: section too short (<4 bars)
```

## Impact

### Files Affected

**New files:**
- `.dvc/` - DVC configuration directory
- `dvc.yaml` - Pipeline definitions
- `dvc.lock` - Lock file for reproducibility
- `data.dvc` - Tracks data/ directory
- `src/edm/data/` - New module for data management
  - `src/edm/data/schema.py` - Annotation schema validation
  - `src/edm/data/metadata.py` - Metadata handling
  - `src/edm/data/export.py` - Export to ML formats
- `src/cli/commands/data.py` - New CLI command

**Modified files:**
- `pyproject.toml` - Add `dvc`, `label-studio-sdk` dependencies
- `.gitignore` - Add DVC entries
- `data/annotations/*.yaml` - Add metadata section (backward compatible)

**Specs affected:**
- `openspec/specs/development-workflow/spec.md` - Add DVC workflow
- New spec: `openspec/specs/data-management/spec.md`

### Breaking Changes

**Complete break with old format:**
- All existing YAML annotations in `data/annotations/` are invalid
- Old format: flat structure with `[[bar, label, time]]` arrays
- New format: structured with `metadata`, `audio`, `structure`, `energy` sections
- Duration format: was `M:SS.mm`, now `float` seconds
- Time signature: was `4/4` string, now `[4, 4]` array
- No raw section comments (replaced by structured energy data)

**Must regenerate all annotations:**
- Re-import from Rekordbox XML using new parser
- Re-analyze with MSAF/energy detectors
- Manually verify high-priority tracks (Tier 1)

### No Migrations

**Clean break approach:**
- Delete `data/annotations/*.yaml` and start fresh
- Existing annotations tracked in git history if needed
- Rekordbox XML is source of truth, can always re-import
- Better to start clean than maintain legacy format

### Dependencies

**Add to pyproject.toml:**
```toml
[project]
dependencies = [
    ...existing...
    "dvc>=3.0.0",  # Data version control
    "dvc-s3",      # S3 remote storage (optional)
]

[project.optional-dependencies]
annotation-ui = [
    "label-studio>=1.10.0",
    "label-studio-sdk>=0.0.32",
]
```

### Risks

**Low risk:**
- DVC well-established, 12k+ stars on GitHub
- Label Studio 17k+ stars, actively maintained
- Both have Python SDKs for programmatic access

**Medium risk:**
- Learning curve for DVC commands
- Label Studio requires running server (can be optional)
- Remote storage setup needed (can use local for now)

**High risk:**
- None identified

**Mitigation:**
- Start with local DVC remote (shared folder)
- Make Label Studio optional (Phase 3)
- YAML format human-readable and git-diffable
- Rekordbox XML remains source of truth (can always re-import)

## Benefits

1. **Versioning:** Track every change to annotations, rollback if needed
2. **Reproducibility:** Know exactly which data version trained which model
3. **Collaboration:** Multiple people can work on annotations safely
4. **Quality tracking:** See dataset evolution over time (confidence scores, tiers)
5. **UI option:** Label Studio for visual annotation review
6. **No reinvention:** Use battle-tested tools instead of custom build
7. **Scalability:** DVC handles large datasets (100GB+) efficiently
8. **Clean structure:** Metadata first-class, not bolted on
9. **ML-ready:** Confidence scores, tiers, validation built into format
10. **Energy integrated:** Multi-band energy data stored with structure

## Drawbacks

1. **Additional tooling:** Need to learn DVC commands
2. **Storage requirement:** Need remote storage (S3, GCS, or local NAS)
3. **Complexity:** More moving parts than simple flat files
4. **Label Studio overhead:** Running server for optional UI (mitigated: make optional)
5. **Must regenerate all data:** Existing annotations become invalid
6. **Format lock-in:** Committing to this structure for ML training

## Comparison with Custom Build

**If we built custom:**
- Estimated 2-3 weeks development time
- Need to handle versioning, conflict resolution, storage
- Need to build UI from scratch
- Need to maintain long-term

**Using existing tools:**
- Setup: 2-3 hours
- Maintained by communities (DVC: 12k stars, Label Studio: 17k stars)
- Battle-tested on large-scale projects
- Free and open-source

**Verdict:** Use existing tools, don't build custom.

## Implementation Order

**Phase 1 (Week 1):** DVC setup + enhanced YAML format
**Phase 2 (Week 2):** Data management CLI (`edm data`)
**Phase 3 (Week 3-4):** Label Studio integration (optional)
**Phase 4 (Ongoing):** Use in MLPIVOT workflow

## Sources

- [DVC - Data Version Control](https://dvc.org/)
- [Label Studio - Open Source Data Labeling](https://labelstud.io/)
- [Label Studio GitHub](https://github.com/HumanSignal/label-studio)
- [Cleanlab - Data-Centric AI](https://pypi.org/project/cleanlab/)
- [Audino - Audio Annotation Tool](https://github.com/midas-research/audino)
- [Top Dataset Version Control Tools](https://www.twine.net/blog/dataset-version-control-tools/)
- [Label Studio for Audio Segmentation](https://labelstud.io/templates/audio_regions)
