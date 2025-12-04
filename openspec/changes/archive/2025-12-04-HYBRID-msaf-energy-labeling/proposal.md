# [HYBRID] Hybrid MSAF-Energy Labeling

**Status**: ready
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

MSAF produces accurate section boundaries but generic cluster-based labels (segment1, segment2) that lack semantic meaning for EDM tracks. Users need EDM-specific labels (intro, buildup, main, breakdown, outro) to understand track structure.

Current labeling approach:
- MSAF: Spectral similarity clustering → segment1, segment2, segment3
- Energy detector: RMS thresholds → drop, breakdown, other (but boundaries less accurate)

## What

Enhance `MSAFDetector` to use energy analysis for EDM-specific labeling while keeping MSAF's superior boundary detection.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - MSAFDetector class
- `openspec/specs/analysis/spec.md` - Structure detection spec

**Changes:**
- Add energy analysis to `MSAFDetector.detect()` after boundary detection
- Calculate per-segment average normalized RMS energy
- Map energy levels to EDM labels using thresholds
- Add position-based heuristics (first→intro, last→outro)

## Impact

**Breaking changes:** None (additive)

**Benefits:**
- Meaningful labels without sacrificing boundary accuracy
- Reuses existing energy calculation logic
- Improves annotation alignment with human expectations

**Risks:**
- ~15-20% computation overhead (RMS calculation)
- Energy thresholds may need per-track calibration
- Mixed-genre tracks may get inappropriate EDM labels

**Mitigation:**
- Make energy labeling optional via parameter
- Add confidence scores to indicate label certainty
- Consider track-adaptive threshold normalization
