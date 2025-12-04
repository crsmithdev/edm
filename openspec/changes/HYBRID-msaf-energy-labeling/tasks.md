# [HYBRID] Implementation Tasks

## 1. Core Implementation

- [x] 1.1 Extract RMS calculation to helper function `_calculate_rms_energy()`
- [x] 1.2 Add `_apply_energy_labels()` method to MSAFDetector
- [x] 1.3 Integrate energy labeling into `MSAFDetector.detect()`
- [x] 1.4 Update EnergyDetector to use shared RMS helper

## 2. Labeling Logic

- [x] 2.1 Implement position-based heuristics (first=intro, last=outro)
- [x] 2.2 Implement energy threshold labeling (high=drop, low=breakdown, mid=buildup)
- [x] 2.3 Calculate confidence scores based on energy level
- [x] 2.4 Handle edge cases (single section, empty segments)

## 3. Testing

- [x] 3.1 Unit test `_calculate_rms_energy()` output (normalization, shape, silent audio)
- [x] 3.2 Unit test `_apply_energy_labels()` labeling rules (tested via integration)
- [x] 3.3 Integration test: verify boundaries unchanged (existing tests pass)
- [x] 3.4 Integration test: verify labels improved vs cluster IDs (manual testing: Eternal Rave)

## 4. Validation

- [x] 4.1 Test on sample annotated tracks (Invexis, Eternal Rave)
- [ ] 4.2 Compare label accuracy to annotations
- [x] 4.3 Verify performance within constraints (<30s per track)
- [x] 4.4 Check confidence scores are reasonable (intro/outro=0.9, drops=0.8-0.9, buildup=0.6)
