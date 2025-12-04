# [ENERGY] Multi-Feature Energy Detection for Structure Analysis

## Problem Statement

Current energy-based structure detection relies solely on RMS energy (structure_detector.py:288), which provides limited information about musical content. This results in:

- **Imprecise drop detection**: RMS alone cannot distinguish between sustained bass drops and other loud sections
- **Poor buildup identification**: Energy rise detection misses timbral and spectral characteristics of buildups
- **Generic labeling**: Cannot capture the rich spectral/timbral differences between EDM section types

MSAF provides boundaries but no EDM-specific labels. The hybrid approach (MSAF boundaries + energy labeling) could significantly improve accuracy if energy analysis incorporated multiple acoustic features.

## Goals

- Enhance energy-based detection with multi-feature analysis (spectral centroid, spectral contrast, band-specific energy)
- Improve drop detection accuracy from current baseline
- Enable hybrid MSAF + multi-feature energy approach
- Maintain <30s/track performance constraint

## Proposed Changes

### 1. Multi-Feature Energy Extraction

Extend `EnergyDetector` to compute multiple features alongside RMS:

**Features to add:**
- **Spectral centroid** (brightness) - drops are typically brighter due to high-frequency content
- **Spectral contrast** (timbral variation) - drops have higher spectral contrast
- **Band-specific energy** (bass/mid/high ratios) - drops have strong bass presence
- **Onset strength** (transient density) - buildups have increasing onset density

**Implementation approach:**
- Extract all features using librosa with consistent hop_length (512 samples)
- Normalize each feature independently
- Apply per-feature thresholding for section classification
- Combine features using weighted scoring or decision rules

### 2. Enhanced Section Labeling

Update `_boundaries_to_sections()` in `EnergyDetector` to use multi-feature scoring:

**Drop criteria:**
- High RMS energy (>0.7 normalized)
- High spectral contrast (>0.6)
- Strong bass energy ratio (>0.5)
- High spectral centroid (bright sound)

**Buildup criteria:**
- Rising RMS gradient
- Increasing onset strength
- Rising spectral centroid
- Mid-high spectral contrast

**Breakdown criteria:**
- Low RMS energy (<0.4)
- Low onset strength
- Lower spectral contrast
- Variable spectral centroid

### 3. Hybrid MSAF + Multi-Feature Energy

Add new detector mode: `hybrid`

**Workflow:**
1. Use MSAF spectral flux for boundary detection (musical precision)
2. Apply multi-feature energy analysis to each segment
3. Assign EDM-specific labels based on feature profiles
4. Return sections with improved boundary placement and accurate labels

### 4. Alternative Libraries (Optional Future Work)

Evaluation options beyond librosa:
- **Essentia**: C++ library with Python bindings, faster feature extraction
- **openSMILE**: Batch feature extraction across multiple files
- **pyAudioAnalysis**: Pre-integrated feature sets

**Decision criteria:**
- Start with librosa (already integrated, no new dependencies)
- Consider alternatives if performance becomes bottleneck or accuracy plateaus

## Files to Modify

### Primary Changes
- `src/edm/analysis/structure_detector.py`:
  - Add feature extraction methods to `EnergyDetector`
  - Update `_boundaries_to_sections()` with multi-feature scoring
  - Add `HybridDetector` class (MSAF boundaries + multi-feature labeling)

- `src/edm/analysis/structure.py`:
  - Add `hybrid` detector option to `analyze_structure()`
  - Update detector selection logic in `get_detector()`

### Testing & Evaluation
- Add test cases for multi-feature extraction
- Extend structure evaluation to compare detector performance

### Documentation
- Update analysis spec with multi-feature energy requirements
- Document feature weights and thresholds

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Drop precision | Baseline | >85% |
| Drop recall | Baseline | >80% |
| Boundary F1 (hybrid mode) | N/A | >65% |
| Analysis time | ~30s | <30s |

## Risks & Trade-offs

### Performance
- Multiple feature extractions increase compute time
- Mitigation: Use consistent hop_length, cache STFT computation

### Complexity
- More features = more parameters to tune
- Mitigation: Start with simple weighted scoring, iterate based on evaluation

### Dependency
- Staying with librosa avoids new dependencies
- Alternative: Essentia would require adding C++ library

### Parameter Tuning
- Feature thresholds may need per-genre tuning
- Mitigation: Use ground truth annotations to optimize thresholds

## Implementation Phases

### Phase 1: Multi-Feature Energy Detector
1. Add spectral centroid, spectral contrast, band energy to `EnergyDetector`
2. Implement multi-feature scoring for section labeling
3. Evaluate on annotated tracks

### Phase 2: Hybrid Detector
1. Create `HybridDetector` using MSAF boundaries
2. Apply multi-feature energy labeling to MSAF segments
3. Evaluate boundary + label accuracy

### Phase 3: Optimization
1. Tune feature weights and thresholds using evaluation metrics
2. Consider alternative libraries if performance bottleneck identified
3. Document optimal parameters

## Related Changes

- **SEGACC**: Complements parameter tuning approach with richer features
- Could be integrated as Phase 2 in SEGACC if parameter tuning yields insufficient gains
