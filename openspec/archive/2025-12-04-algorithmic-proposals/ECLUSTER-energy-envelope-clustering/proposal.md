# [ECLUSTER] Energy Envelope Clustering

**Status**: draft
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

Current energy-based labeling uses fixed thresholds (high energy=drop, low energy=breakdown). This fails for:
- Tracks with non-standard loudness (quiet drops, loud breakdowns)
- Gradual energy changes (buildups vs sustained mid-energy sections)
- Genre variations (dubstep drops ≠ trance drops)

Fixed thresholds at `src/edm/analysis/structure_detector.py:405-416`:
```python
if avg_energy > 0.7:      # drop
elif avg_energy < 0.4:    # breakdown
else:                      # other
```

Better: cluster energy *shapes* (rising, falling, flat-high, flat-low) rather than absolute levels.

## What

Replace threshold-based labeling with k-means clustering of energy envelope features.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - EnergyDetector labeling logic
- `openspec/specs/analysis/spec.md` - Structure detection spec

**Approach:**
1. For each detected section, extract energy envelope features:
   - Average level (normalized)
   - Slope (rising/falling trend)
   - Variance (stable vs dynamic)
   - Peak-to-average ratio
2. Cluster sections into k=5 groups using k-means
3. Map clusters to EDM labels based on feature signatures:
   - Rising slope → buildup
   - High-flat → drop
   - Low-flat → breakdown
   - Falling → outro
   - Mixed → intro/other

**Configuration:**
- Number of clusters (default: 5)
- Feature weights
- Cluster-to-label mapping heuristics

## Impact

**Breaking changes:** None (alternative labeling strategy)

**Benefits:**
- Track-adaptive (learns from actual energy distribution)
- Captures temporal patterns (rising/falling)
- Reduces manual threshold tuning
- Better cross-genre generalization

**Risks:**
- K-means requires sufficient sections (min 10-15 for k=5)
- Cluster assignment may be unstable for edge cases
- Less interpretable than explicit thresholds
- May need track-specific k tuning

**Computational cost:**
- K-means on ~10-20 sections: negligible (<0.1s)
- Feature extraction: reuses existing RMS calculation

**Mitigation:**
- Fall back to threshold-based if too few sections
- Add cluster stability metrics to confidence scores
- Provide debug output showing cluster assignments
- Make k configurable per track
