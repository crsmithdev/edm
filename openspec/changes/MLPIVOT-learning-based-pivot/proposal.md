---
status: draft
created: 2025-12-04
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
- `data/annotations/` - Extend schema for energy metrics and feature vectors

**New files:**
- `src/edm/models/backbone.py` - MERT/wav2vec embedding extraction
- `src/edm/models/heads.py` - Task-specific prediction heads
- `src/edm/training/dataset.py` - Dataset class for DJ software labels
- `src/edm/training/losses.py` - Boundary-tolerant loss functions
- `scripts/train.py` - Training entry point
- `scripts/validate_labels.py` - Cleanlab integration for error detection

### Specs Affected
- `openspec/specs/analysis/spec.md` - New ML-based detection modes
- `openspec/specs/development-workflow/spec.md` - Training workflow

## Impact

### Breaking Changes
Breaking changes are acceptable and expected:
- ML detector as primary, algorithmic detectors may be removed if not needed
- Training data format and annotation schema will be redesigned
- Existing code/workflows optimized for rule-based approaches can be replaced

### Migrations
No migrations - clean break approach:
- Existing algorithms can be completely replaced by ML equivalents
- No requirement to maintain backward compatibility with old annotation formats
- Fresh start on data pipeline and model architecture

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

1. **Cue point format**: What format are cue points in? (Rekordbox XML, Serato, Traktor NML, other?)
2. **Beat grid source**: Are beat grids from metadata or computed? Any systematic errors (double/half time)?
3. **Clean subset**: Can you provide ~200 tracks with manually verified structure annotations?
4. **Energy quantification**: What energy representation do you want?
   - Continuous 0-1 per-frame
   - Per-section average
   - Discrete levels (low/medium/high)
   - Multiple dimensions (bass energy, mid energy, high energy)
5. **Hardware**: GPU available for training? CUDA/MPS/CPU-only?
6. **Feature priorities**: Rank these by importance:
   - Segment boundary detection
   - Energy quantification
   - Beat/downbeat tracking improvement
   - Key detection
