# Design: [XVAL] Cross-Validation Framework

## Context

**Problem**: The rekordbox importer bug produced 351/356 annotations with incorrect bar indexing (all bars at 1). This would have poisoned ML training if not caught. We need automated validation to:

1. **Validate training data** - Catch systematic errors before they enter the pipeline
2. **Validate ML predictions** - Cross-check model outputs against beat grid
3. **Diagnose error sources** - Determine if errors are in BPM, structure, downbeat, or phase

**Current Architecture:**
- **BPM**: Neural beat_this model (fallback to librosa) - highly accurate ML
- **Structure**: MSAF (spectral) + energy-based labeling - ML-informed
- **Both run independently** - no cross-validation between signals

When ML-predicted structure boundaries don't align with the beat grid, it indicates error, but we don't know which source is wrong.

## Goals / Non-Goals

**Goals:**
- Validate annotation quality before ML training (prevent garbage in)
- Cross-validate ML predictions against beat grid alignment (detect model errors)
- Pluggable validator architecture for future signal types (energy, onset strength)
- Detect common error patterns (BPM offset, phase error, converter bugs, drift)
- Report alignment metrics with confidence-weighted arbitration
- Start with flag-only mode, enable corrections after validation

**Non-Goals:**
- Auto-correct in initial release (flagging only - manual review required)
- Variable tempo support (EDM has constant tempo)
- Real-time validation during analysis (post-analysis only)
- Replace existing BPM or structure detectors (validation layer, not replacement)

## Decisions

### Validator Protocol

Validators implement a simple protocol:

```python
class Validator(Protocol):
    name: str

    def validate(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float = 0.25,  # 1/16 bar
    ) -> ValidationResult: ...

    def is_applicable(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
    ) -> bool: ...
```

**Rationale:** Simple interface allows adding validators without modifying orchestrator. Future validators (key/harmony, tempo/energy) follow same pattern.

### Error Pattern Classification

| Pattern | Detection Criteria | Meaning |
|---------|-------------------|---------|
| `BPM_SYSTEMATIC_OFFSET` | Low std dev, consistent offset | BPM slightly wrong (rare with beat_this) |
| `DOWNBEAT_PHASE_ERROR` | Offset ~0.5 bars consistently | Wrong downbeat phase |
| `TIME_SIGNATURE_MISMATCH` | Offset ~0.25 bars consistently | Wrong time signature assumption |
| `SINGLE_BOUNDARY_ERROR` | One boundary off, others aligned | ML structure prediction error |
| `PROGRESSIVE_DRIFT` | Error increases over time | BPM accumulation or tempo variation |
| `CONVERTER_BUG` | All bars at 1 or constant offset | Data import/conversion error |

**Rationale:** Based on research, DJ software patterns (Mixxx), and real bugs found in this project (rekordbox converter). Patterns help diagnose whether errors are in ML models, algorithmic components, or data pipelines.

### Post-Analysis Integration

Validation runs after both BPM and structure analysis complete:

```
analyze_bpm() ──┐
                ├──> validate() ──> report
analyze_structure()
```

**Rationale:** Simpler than joint estimation, maintains backward compatibility, corrections can be added later without restructuring.

### Tolerance Default

Default tolerance: **0.25 beats** (1/16 bar in 4/4)

**Rationale:** EDM drops should hit exact bar lines. Stricter than Mixxx's 25ms (which is ~0.05 beats at 128 BPM) but allows for minor detection variance.

## Module Structure

```
src/edm/analysis/validation/
├── __init__.py          # Public API: validate_analysis()
├── base.py              # Validator protocol, BaseValidator
├── beat_structure.py    # BeatStructureValidator
├── orchestrator.py      # ValidationOrchestrator
└── results.py           # Dataclasses: ErrorPattern, AlignmentError, etc.
```

## ML-Specific Use Cases

### Training Data Validation

**Problem**: Annotations from various sources (manual, rekordbox, algorithmic) may contain systematic errors.

**Solution**: Validate all annotations before adding to training set:

```bash
# Validate before training
edm data validate data/annotations/ --strict

# Filter by validation results in dataset
dataset = EDMDataset(
    annotation_dir="data/annotations",
    tier_filter=1,  # Only tier-1
    min_confidence=0.9,  # High confidence
    require_valid=True  # Only validation-passed samples
)
```

**Impact**: Prevents garbage data from poisoning model training, improves generalization.

### ML Prediction Confidence

**Problem**: ML models output confidence scores, but don't account for physical constraints (bar alignment).

**Solution**: Cross-validate predictions, adjust confidence based on alignment:

```python
# After structure prediction
structure_result = analyze_structure(audio_path, detector="msaf")
validation_result = validate_analysis(bpm_result, structure_result)

# Adjust confidence based on alignment
if validation_result.alignment_quality < 0.7:
    # Penalize confidence for misaligned predictions
    adjusted_confidence = structure_result.confidence * 0.5
else:
    adjusted_confidence = structure_result.confidence
```

**Impact**: More calibrated confidence scores, better sample weighting during training.

### Error Attribution

**Problem**: When predictions are wrong, need to know if it's BPM error, structure error, or both.

**Solution**: Validation patterns indicate error source:

- `CONVERTER_BUG` → Data pipeline issue, retrain not needed
- `SINGLE_BOUNDARY_ERROR` → ML structure model error, needs more training data
- `DOWNBEAT_PHASE_ERROR` → BPM detection phase error, not structure issue
- `BPM_SYSTEMATIC_OFFSET` → BPM model error (rare with beat_this)

**Impact**: Targeted improvements - don't retrain structure model for BPM errors.

## Cross-Domain Best Practices

Patterns from sensor fusion, data engineering, ensemble ML, and distributed systems inform this design.

### 1. Confidence-Weighted Arbitration

From ensemble ML: disagreement between sources = uncertainty. Rather than guessing when signals conflict, weight by confidence:

```python
if bpm_confidence > structure_confidence + 0.2:
    # Trust BPM, suggest quantizing structure boundaries
    action = "quantize_structure"
elif structure_confidence > bpm_confidence + 0.2:
    # Trust structure, suggest BPM correction
    action = "correct_bpm"
else:
    # Genuine ambiguity - flag for review
    action = "flag_conflict"
```

**Rationale:** With only 2 signals, we can detect conflict but can't arbitrate without confidence weighting. BFT requires N=3F+1 to tolerate F faults.

### 2. Uncertainty vs Confidence

Report both separately:
- **Confidence**: How sure is each detector of its own result?
- **Uncertainty**: How much do the signals disagree?

High confidence + high uncertainty = conflicting signals (needs review).
Low confidence + low uncertainty = weak but consistent (proceed cautiously).

### 3. Cascading Validation Levels

From data reconciliation: tradeoff between granularity and resources.

| Level | Check | Cost | When |
|-------|-------|------|------|
| Quick | Any boundary >0.5 bars off? | O(n) | Always |
| Standard | Pattern detection, mean/std | O(n) | Default |
| Deep | All pairwise, drift analysis | O(n²) | On request |

### 4. Fault Detection Before Fusion (FDE)

From sensor fusion: detect bad signal *before* applying corrections. Validate each source independently:

```python
# Check BPM plausibility
bpm_valid = 40 <= bpm <= 200 and bpm_confidence > 0.3

# Check structure plausibility
structure_valid = len(sections) >= 2 and all(s.confidence > 0.3 for s in sections)

# Only cross-validate if both pass basic checks
if bpm_valid and structure_valid:
    cross_validate()
```

### 5. Consistency Matrix (Future: 3+ Signals)

When we add more signals (beat positions, energy peaks, onset strength), use pairwise consistency matrix. A "cross pattern" (one row/column all high disagreement) isolates the faulty source.

```
         BPM   Structure   Beats   Energy
BPM       -      0.8       0.1     0.2
Structure 0.8     -        0.9     0.7    ← Structure disagrees with all
Beats     0.1    0.9        -      0.2
Energy    0.2    0.7       0.2      -
```

### 6. Adaptive Trust (Future)

Track historical accuracy per detector method. Adjust weights based on empirical performance against ground truth evaluations.

```python
# After evaluation against ground truth
detector_accuracy["beat_this"] = 0.95
detector_accuracy["librosa"] = 0.82
detector_accuracy["msaf"] = 0.78
detector_accuracy["energy"] = 0.71

# Use in future validations
bpm_weight = detector_accuracy[bpm_result.method]
```

### 7. Data Contracts Pattern

Validators are essentially data contracts - define what "valid" analysis looks like:

- Boundaries should fall on whole bars (±tolerance)
- BPM should be in valid range (40-200)
- Sections should cover full track with no gaps
- Confidence scores should be calibrated (not always 0.99)

## Module Structure

```
src/edm/analysis/validation/
├── __init__.py          # Public API: validate_analysis()
├── base.py              # Validator protocol, BaseValidator
├── beat_structure.py    # BeatStructureValidator
├── orchestrator.py      # ValidationOrchestrator
└── results.py           # Dataclasses: ErrorPattern, AlignmentError, etc.
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Over-flagging correct analyses | Conservative thresholds, require pattern confidence >0.7 |
| False error diagnosis | Start with flagging only, manual review before auto-correct |
| Performance overhead | Validation is O(n) where n = boundaries, negligible vs analysis |
| Two-signal limitation | Can detect conflict but not arbitrate; use confidence weighting |
| Overconfident detectors | Track calibration; flag when confidence high but signals disagree |

## Open Questions

1. Should validation run by default or require `--validate`?
   - **Recommendation:** Auto-trigger when confidence < 0.8, explicit flag forces on/off

2. Output format for validation results?
   - **Recommendation:** Nested JSON under `"validation"` key, human-readable summary in text mode

3. Which third signal to add first for better arbitration?
   - **Candidates:** Beat positions (from beat tracker), energy peaks, onset strength at boundaries
   - **Recommendation:** Defer to phase 2; start with confidence-weighted arbitration
