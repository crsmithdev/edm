# [SEGACC] Improve Structure Segmentation Accuracy

## Problem Statement

Current MSAF-based structure detection achieves only 35.5% boundary F1 score across annotated EDM tracks, with high variance (0%-83% per track). The spectral flux algorithm with default parameters fails to find boundaries that match human perception of EDM structure.

## Goals

- Improve boundary F1 from 35.5% to >60%
- Reduce variance across tracks
- Maintain <30s/track performance constraint

## Research Findings

### MSAF Algorithm Options
- **sf (spectral flux)**: Current default, 35.5% F1
- **foote**: Checkerboard kernel, larger smoothing (M_gaussian=66)
- **cnmf**: Matrix factorization, may capture repetition better
- **scluster**: Spectral clustering, multi-layer approach

### Tunable Parameters for SF
```
M_gaussian: 27     # SSM smoothing kernel size
m_embedded: 3      # Embedding dimension
k_nearest: 0.04    # KNN ratio for affinity
Mp_adaptive: 28    # Adaptive threshold window
offset_thres: 0.05 # Peak picking threshold
```

### Alternative Approaches
- **all-in-one**: Neural network, trained on Harmonix, 10 EDM labels
- **MusicBoundariesCNN**: CNN on SSLM + spectrograms
- **Hybrid**: MSAF candidates + energy validation

## Proposed Approach

### Phase 1: Parameter Tuning (Iterative)
Test each algorithm and parameter combination:

1. **Algorithm comparison**: sf vs foote vs cnmf vs scluster
2. **SF parameter sweep**:
   - offset_thres: [0.02, 0.05, 0.08, 0.1]
   - M_gaussian: [15, 27, 40, 60]
3. **Best algorithm parameter tuning**

Track F1 scores per iteration to find optimal config.

### Phase 2: Hybrid Validation
If Phase 1 < 60% F1:
- Use best MSAF config for candidate boundaries
- Filter with energy-based confirmation
- Accept only boundaries with significant RMS/spectral change

### Phase 3: Alternative Detector Evaluation
If Phase 2 < 60% F1:
- Integrate mir-aidj/all-in-one as detector option
- Compare on same test set
- Use whichever performs better

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Boundary F1 | 35.5% | >60% |
| Per-track variance | 0-83% | <30% spread |
| Analysis time | ~30s | <30s |

## Files to Modify

- `src/edm/analysis/structure_detector.py`: MSAF config, new algorithms
- `src/edm/analysis/structure.py`: Algorithm selection logic
- `.claude/scripts/evaluate.sh`: Iteration tracking

## Risks

- Parameter tuning may yield marginal gains
- EDM structure may need genre-specific tuning
- Alternative detectors add dependencies
