# [XVAL] Add Cross-Validation Framework

status: ready

## Why

**Training Data Quality Crisis**: The rekordbox importer had a bar indexing bug that produced 351/356 broken annotations (bars all at 1, causing structure boundaries to misalign with the beat grid). Without automated validation, these would poison ML training. Cross-validation detects such errors automatically.

**ML Inference Validation**: Structure detection uses ML models (MSAF + energy-based labeling). When ML-predicted boundaries don't align with the neural BPM detector (beat_this), it indicates either model error, BPM error, or phase error. We need systematic validation to:

1. **Clean training data** - Validate annotations before training, flag systematic errors
2. **Validate ML predictions** - Cross-check model outputs against beat grid alignment
3. **Assign confidence scores** - Use alignment quality to weight training samples
4. **Debug error sources** - Diagnose whether errors are in BPM, structure, downbeat, or phase

Research from [ISMIR/TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.167) shows downbeat alignment improved structure accuracy from 27% to 43%.

**Current State**: BPM uses neural beat_this model, structure uses MSAF + heuristics or energy detector. Both are ML-informed but run independently. The rekordbox converter bug (now fixed) showed we need validation before data enters the training pipeline.

## What Changes

**Core Validation Framework:**
- Add pluggable validator framework (`src/edm/analysis/validation/`)
- Implement beat/structure alignment validator as first validator
- Implement downbeat/structure alignment validator using `BeatGrid`
- Cross-validate four independent signals:
  - `ComputedBPM` from beat_this (neural ML model)
  - `BeatGrid` - beat positions (index-based)
  - `BeatGrid.first_downbeat` - downbeat phase
  - Structure sections - ML-predicted or algorithmic

**Integration Points:**
- Add `edm data validate` command - validate annotation quality before training
- Add `--validate` flag to `edm analyze` - validate predictions during inference
- Integrate with `EDMDataset` - filter/flag low-quality samples based on validation
- Report alignment metrics and error patterns in analysis output

**Phased Rollout:**
- Phase 1: Flag-only mode, report alignment issues
- Phase 2: Auto-correction for high-confidence cases
- Phase 3: Confidence-weighted sample filtering for training

## Impact

- **Effort**: 5 days
- **ROI**: Critical - prevents training on bad data, validates ML predictions
- **Dependencies**: None (pure Python, uses existing analysis results)
- **Affected specs**: `analysis` (adds cross-validation), `data-management` (validation integration)
- **Affected code**:
  - `src/edm/analysis/validation/` (new module)
  - `src/cli/commands/analyze.py` (--validate flag)
  - `src/cli/commands/data.py` (integrate with data validate command)
  - `src/edm/training/dataset.py` (optional: filter by validation results)
