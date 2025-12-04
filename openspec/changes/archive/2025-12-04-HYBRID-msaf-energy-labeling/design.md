# [HYBRID] Design

## Approach

Keep MSAF boundary detection, add energy-based EDM labeling as post-process.

**Flow:**
1. MSAF detects boundaries → segments with cluster labels
2. Load audio, calculate RMS energy curve
3. For each segment, calculate average normalized energy
4. Apply labeling rules: energy thresholds + position heuristics
5. Return segments with EDM labels + confidence scores

## Implementation

### `src/edm/analysis/structure_detector.py`

**MSAFDetector.detect()**
- After line 123 (boundaries created), add energy labeling:
  ```python
  # Apply EDM labeling based on energy
  sections = self._apply_energy_labels(sections, y, sr)
  ```

**New method: MSAFDetector._apply_energy_labels()**
- Calculate RMS energy curve (reuse EnergyDetector logic)
- For each section:
  - Get average energy in segment
  - Apply labeling rules
- Return sections with updated labels

**Labeling rules:**
```python
if position == first:
    label = "intro"
elif position == last:
    label = "outro"
elif avg_energy > 0.7:
    label = "main"  # High-energy main sections (drops are events, not sections)
    confidence = 0.8 + (avg_energy - 0.7) * 0.5
elif avg_energy < 0.4:
    label = "breakdown"
    confidence = 0.75
else:
    label = "buildup"
    confidence = 0.6
```

**Reuse energy calculation:**
- Extract RMS calculation from `EnergyDetector.detect()` lines 286-296
- Move to helper function: `_calculate_rms_energy(y, sr) -> np.ndarray`
- Both MSAFDetector and EnergyDetector call it

## Testing

1. Unit tests:
   - `_calculate_rms_energy()` produces correct shape/values
   - `_apply_energy_labels()` assigns expected labels
   - Edge cases: single section, very short track

2. Integration tests:
   - Compare MSAF with/without energy labeling
   - Verify boundaries unchanged, only labels change
   - Check confidence scores in valid range

3. Evaluation:
   - Run on annotated tracks
   - Measure label accuracy vs annotations
   - Compare to pure MSAF and pure Energy detector

## Risks

- **Energy thresholds**: May need tuning per subgenre
  - Mitigation: Make thresholds configurable
- **Computation overhead**: RMS adds ~15-20%
  - Acceptable within 30s constraint
- **Label conflicts**: Energy may conflict with spectral patterns
  - Use confidence scores to indicate uncertainty
