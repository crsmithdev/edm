# [TEMPORAL] Temporal Energy Context Labeling

**Status**: draft
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

Current labeling considers sections independently, missing contextual EDM structure patterns:
- Low energy after high energy → typically **breakdown**
- Low energy before high energy → typically **buildup**
- High energy in middle third → typically **drop**
- First/last sections → intro/outro (already implemented)

Example failure case:
```
Section 1: mid-energy → labeled "other"
Section 2: mid-energy → labeled "other"
Section 3: high-energy → labeled "drop"
```

With context, Section 2 would be labeled "buildup" (rising toward drop).

## What

Add context-aware labeling pass that considers section position and adjacent energy levels.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - Add `_apply_temporal_context()` function
- `src/edm/analysis/structure.py` - Call context refinement after detection
- `openspec/specs/analysis/spec.md` - Document context rules

**Approach:**
1. Initial labeling: Use existing energy/MSAF detector
2. Context refinement pass:
   ```python
   for i, section in enumerate(sections):
       prev_energy = sections[i-1].avg_energy if i > 0 else None
       next_energy = sections[i+1].avg_energy if i < len-1 else None
       position = i / len  # 0.0=start, 1.0=end

       # Apply heuristics:
       if prev_energy > section.avg_energy > next_energy:
           label = "breakdown"  # energy dip
       elif prev_energy < section.avg_energy < next_energy:
           label = "buildup"    # energy rise
       elif position > 0.3 and position < 0.7 and high_energy:
           label = "drop"       # middle-section peak
   ```

**Rules:**
- Energy drop (prev > curr): breakdown candidate
- Energy rise (curr > next): buildup candidate
- Middle position + high energy: drop reinforcement
- Preserve high-confidence labels from initial detection

## Impact

**Breaking changes:** None (refinement layer)

**Benefits:**
- Improves buildup detection (currently often labeled "other")
- Disambiguates mid-energy sections
- Matches human structural understanding
- Minimal computational cost

**Risks:**
- Over-fitting to common EDM patterns
- May misclassify non-standard structures
- Conflicts with initial detector labels

**Mitigation:**
- Only apply context when initial confidence <0.8
- Make context rules configurable
- Add bypass flag for non-EDM tracks
- Weight context heuristics lower than detector output

**Performance:**
- O(n) pass over sections (negligible cost)
