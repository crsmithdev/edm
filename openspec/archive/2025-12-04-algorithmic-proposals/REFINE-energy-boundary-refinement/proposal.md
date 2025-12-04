# [REFINE] Energy-Assisted Boundary Refinement

**Status**: draft
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

MSAF detects boundaries based on spectral change, which may precede or follow actual drop moments. Example:

```
Time:      60.0s    61.5s    62.0s    63.0s
MSAF:               ^boundary
Energy:                      ^peak
Actual drop:                 ^62.0s
```

MSAF detects spectral shift at 61.5s, but actual drop impact is at 62.0s (energy peak). Annotations typically mark the energy peak, not the spectral change.

Current mismatch affects:
- Annotation alignment evaluation
- User-facing timestamps
- Beat grid synchronization

## What

Add post-processing pass to snap MSAF boundaries to nearest local energy maxima/minima.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - Add boundary refinement function
- `src/edm/analysis/structure.py` - Apply refinement after MSAF detection
- `openspec/specs/analysis/spec.md` - Document refinement algorithm

**Approach:**
1. MSAF produces initial boundaries
2. Calculate RMS energy curve (same as EnergyDetector)
3. For each boundary at time `t`:
   - Search window: `[t-2.0s, t+2.0s]`
   - Find local energy maximum (for drop boundaries)
   - Find local energy minimum (for breakdown boundaries)
   - Determine type by comparing energy before/after boundary
4. Snap boundary to energy extremum if:
   - Extremum confidence >0.7
   - Shift distance <2.0s
   - Doesn't create section overlap

**Configuration:**
- Search window size (default: ±2.0s)
- Max shift distance (default: 2.0s)
- Enable/disable per boundary type

## Impact

**Breaking changes:** None (optional refinement)

**Benefits:**
- Better alignment with human annotations
- Drop boundaries hit actual drop moments
- Improved segmentation accuracy metrics

**Risks:**
- May incorrectly snap boundaries for non-standard structures
- Could introduce overlaps if not careful
- Energy peaks may not always indicate structural change
- Adds computation overhead

**Performance:**
- Energy calculation: already done in hybrid labeling
- Local extremum search: O(n) per boundary, ~0.1s total
- Negligible if energy already computed

**Mitigation:**
- Make refinement optional (disabled by default initially)
- Add confidence threshold for snapping
- Validate no overlaps introduced
- Log boundary shifts for debugging

**Interaction with other proposals:**
- Synergizes with [HYBRID]: reuses energy calculation
- Conflicts with [BEATSYNC]: both adjust boundaries differently
- Consider: apply BEATSYNC first, then REFINE
