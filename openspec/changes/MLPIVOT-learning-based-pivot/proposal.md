---
status: in-progress
created: 2025-12-04
started: 2025-12-04
progress: phases 1-5 complete (data pipeline, model, training, label quality)
next: phase 6 (inference integration), phase 7 (evaluation)
---

# [MLPIVOT] Pivot to Learning-Based Audio Analysis

## Why

### Current Limitations
- Algorithmic approaches (MSAF, energy thresholds) are brittle and don't generalize well
- Rule-based labeling (energy > 0.7 = "main") misses nuanced structure
- No way to improve accuracy without manual algorithm tuning
- Current structure detection produces generic "segment1, segment2" labels that require manual refinement

### Opportunity
- Access to hundreds to thousands of tracks with semi-labeled data from DJ software:
  - Beat grids (high accuracy but with some errors)
  - Key detection
  - BPM (reliable)
  - Cue points that align with segment boundaries
- Learning-based approaches can:
  - Automatically improve with more data
  - Capture patterns too complex for rules
  - Generalize across EDM subgenres
  - Leverage relationships between signals (beat + energy + spectral = structure)

### Shift in Focus
- **From**: Naming segments (intro, buildup, drop)
- **To**: Quantifying characteristics (energy, density, tension, spectral brightness)
- The real value is in relationships between signals, not categorical labels

## What

### Files Affected

**New directories:**
- `src/edm/models/` - Model management, loading, inference
- `src/edm/features/` - Feature extraction pipeline
- `src/edm/training/` - Training loop, data loading, loss functions

**Modify:**
- `src/edm/analysis/structure_detector.py` - Add `MLDetector` implementing `StructureDetector` protocol
- `src/edm/analysis/structure.py` - Wire in ML detector as primary

**New files:**
- `src/edm/models/backbone.py` - MERT/wav2vec embedding extraction
- `src/edm/models/heads.py` - Task-specific prediction heads
- `src/edm/training/dataset.py` - Dataset class for DJ software labels (uses `Annotation` schema from `src/edm/data/schema.py`)
- `src/edm/training/losses.py` - Boundary-tolerant loss functions
- `scripts/train.py` - Training entry point
- `scripts/validate_labels.py` - Cleanlab integration for error detection

**Leverage existing DATAMGMT infrastructure:**
- `src/edm/data/schema.py` - Already has `Annotation`, `EnergyData`, metadata fields
- `src/cli/commands/data.py` - Use `edm data stats`, `edm data validate`, `edm data export` for dataset management
- DVC already configured (`.dvc/`, `dvc.yaml`, `data.dvc`) for versioning training data

### Specs Affected
- `openspec/specs/analysis/spec.md` - New ML-based detection modes
- `openspec/specs/development-workflow/spec.md` - Training workflow, DVC integration

## Impact

### Dependencies

**Already added by DATAMGMT:**
- `pydantic>=2.0.0` - Schema validation for annotations
- `dvc>=3.0.0` - Dataset versioning

**Need to add for ML training:**
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - MERT/wav2vec pretrained models
- `cleanlab>=2.0.0` - Noisy label detection
- `tensorboard` - Training visualization
- `torchaudio>=2.0.0` - Audio preprocessing

**Optional for advanced features:**
- `wandb` - Experiment tracking (alternative to tensorboard)
- `torch-audiomentations` - Data augmentation
- `onnx` - Model export for production inference

### Breaking Changes
Breaking changes are acceptable and expected:
- ML detector as primary, algorithmic detectors may be removed if not needed
- Annotation schema already redesigned by DATAMGMT (no further changes needed)
- Existing code/workflows optimized for rule-based approaches can be replaced

### Migrations
No migrations - clean break approach:
- Existing algorithms can be completely replaced by ML equivalents
- DATAMGMT already handled annotation format migration (clean slate)
- Fresh start on model architecture and training pipeline

### Risks
- Training data quality: DJ software labels have errors
  - Mitigation: Cleanlab for error detection, clean validation set
- Dataset size: Hundreds to thousands may be insufficient for training from scratch
  - Mitigation: Transfer learning from pretrained music models (MERT, beat_this)
- Computational: Training requires GPU
  - Mitigation: Use pretrained backbones, fine-tune only heads
- Overfitting on EDM subgenres in training set
  - Mitigation: Data augmentation, dropout, held-out genre testing

## Benefits and Drawbacks

### Benefits
1. **Accuracy improvement**: Models can capture subtle patterns rules miss
2. **Automatic scaling**: More data = better model, no manual tuning
3. **Multi-task synergy**: Joint learning of beat/structure/energy improves all tasks
4. **Feature relationships**: Network learns how signals combine
5. **Quantitative focus**: Energy regression instead of categorical labels
6. **Transfer learning**: Pretrained music models reduce data requirements to 300-500 tracks for fine-tuning
7. **Data infrastructure ready**: DATAMGMT provides schema, DVC versioning, CLI tools, validation

### Drawbacks
1. **Black box**: Harder to debug than rule-based
2. **Data dependency**: Performance tied to training data quality
3. **Computational cost**: Inference ~10x slower than rules (but still <1s/track with GPU)
4. **Initial complexity**: Training pipeline, model management, versioning
5. **Requires curation**: Someone must validate/correct labels

## Training Data Requirements

### Minimum for Transfer Learning
Based on research (see `docs/learning-based-pivot-research.md`):

| Approach | Minimum Tracks | Notes |
|----------|----------------|-------|
| Fine-tune pretrained backbone | 300-500 | Lowest risk, recommended start |
| Multi-task from scratch | 1,000-2,000 | Higher accuracy potential |
| Semi-supervised expansion | 200 labeled + 2,000 unlabeled | Teacher-student training |

### Data Quality Strategy
1. **Tier 1 (Clean)**: 200-500 tracks manually verified - gold standard for validation
2. **Tier 2 (Noisy)**: 1,000+ tracks from DJ software - used with noise-robust training
3. **Tier 3 (Pseudo)**: Unlimited unlabeled - teacher-student semi-supervised learning

### Handling Label Errors

**Detection:**
- Cleanlab for confident learning (no hyperparameters, works with any model)
- Heuristics: cue points not on beat boundaries = likely error
- Cross-reference: multiple DJ software should agree on boundaries

**Training Strategies:**
- Co-teaching: Train two networks, each teaches the other on samples where they agree
- Loss reweighting: Down-weight high-loss samples (likely errors)
- Label smoothing: Soften hard labels to reduce memorization
- Boundary tolerance: Loss tolerant to ±N frames (beat_this approach)

**Correction Loop:**
1. Train initial model on noisy data
2. Run cleanlab to identify likely errors
3. Manually correct highest-confidence errors
4. Retrain and repeat

## Questions for User

**Data-related questions answered by DATAMGMT:**
- ✅ Energy format: Multi-band (bass/mid/high) per-section + overall, implemented in `EnergyData` schema
- ✅ Annotation format: Structured YAML with metadata (tier, confidence, source) via `Annotation` schema
- ✅ Versioning: DVC configured for tracking training datasets

**ML-specific decisions:**

1. ✅ **Cue point format**: Rekordbox XML
2. ✅ **Beat grid source**: DJ software metadata (from Rekordbox analysis)
3. ✅ **Training data**: 10-20 manually verified (Tier 1) + 500 slightly noisy (Tier 2)
4. ✅ **Hardware**: CUDA GPU available
5. ✅ **Model priorities** (ranked):
   1. Segment boundary detection (core structural analysis)
   2. Beat/downbeat tracking improvement
   3. Energy quantification per section (secondary)
   4. Section label classification (optional)
6. ✅ **Training approach**: Fine-tune pretrained MERT backbone on 500 tracks (lowest risk, sufficient data)

**Implementation implications:**
- Phase 1.1.1: Implement Rekordbox XML parser for cue points and beat grids
- Phase 2.1.3: CUDA device handling and GPU optimization
- Phase 3.1: Focus on boundary head and beat head first, energy head secondary
- Phase 4.1: Dataset handles 10-20 verified for validation, 500 noisy for training
- Phase 4.2: Boundary-tolerant loss with cleanlab filtering for noisy labels
- Fine-tuning strategy: Freeze MERT base, train last 2-3 layers + prediction heads

## Related Proposals

**Dependencies:**
- **[DATAMGMT] Data Management Overhaul** (implemented) - Provides annotation schema, DVC versioning, metadata tracking (tier/confidence/source), `edm data` CLI commands. MLPIVOT leverages this infrastructure for training data management.

**Supersedes:**
- 8 algorithmic proposals archived by [CLEANUP] (2025-12-04): BEATSYNC, ECLUSTER, ENERGY, MULTIENG, REFINE, TEMPORAL, SEGACC, HYBRID. ML models learn these patterns instead of hand-coding rules.

**Workflow:**
1. DATAMGMT set up data infrastructure (complete)
2. MLPIVOT implements ML training pipeline (this proposal)
3. Use `edm data` commands to manage training datasets
4. Use DVC to version datasets and track which data trained which model
