# [ENERGY] Implementation Tasks

## 1. Feature Extraction Infrastructure

- [ ] 1.1 Add `_extract_features()` method to `EnergyDetector` in `src/edm/analysis/structure_detector.py`
  - [ ] 1.1.1 Extract RMS energy (already implemented, refactor to use new method)
  - [ ] 1.1.2 Extract spectral centroid using `librosa.feature.spectral_centroid()`
  - [ ] 1.1.3 Extract spectral contrast using `librosa.feature.spectral_contrast()` (mean across bands)
  - [ ] 1.1.4 Extract onset strength using `librosa.onset.onset_strength()`
  - [ ] 1.1.5 Compute STFT for band-specific energy analysis
  - [ ] 1.1.6 Calculate bass/mid/high energy ratios from STFT
  - [ ] 1.1.7 Apply median filtering (size=21) to all features
  - [ ] 1.1.8 Normalize all features to [0, 1] range

- [ ] 1.2 Add `_normalize()` helper method for [0, 1] normalization

- [ ] 1.3 Update `EnergyDetector.detect()` to call `_extract_features()` instead of inline RMS calculation

## 2. Multi-Feature Section Scoring

- [ ] 2.1 Add `_score_segment()` method implementing multi-feature scoring logic
  - [ ] 2.1.1 Implement drop scoring (RMS + bass + contrast + centroid + onset)
  - [ ] 2.1.2 Implement breakdown scoring (inverted RMS + onset + bass)
  - [ ] 2.1.3 Keep intro/outro position-based heuristics
  - [ ] 2.1.4 Return label and confidence score

- [ ] 2.2 Create `_boundaries_to_sections_multifeature()` method
  - [ ] 2.2.1 Iterate through boundaries
  - [ ] 2.2.2 Calculate average feature values per segment
  - [ ] 2.2.3 Call `_score_segment()` for each section
  - [ ] 2.2.4 Return list of `DetectedSection` with labels and confidence

- [ ] 2.3 Update `EnergyDetector.detect()` to use `_boundaries_to_sections_multifeature()`

- [ ] 2.4 Keep `_detect_boundaries()` unchanged (gradient-based boundary detection still valid)

## 3. Hybrid Detector Implementation

- [ ] 3.1 Create `HybridDetector` class in `src/edm/analysis/structure_detector.py`
  - [ ] 3.1.1 Initialize with `MSAFDetector` and `EnergyDetector` instances
  - [ ] 3.1.2 Implement `detect()` method: get MSAF boundaries, extract features, label segments
  - [ ] 3.1.3 Extract boundary times from MSAF sections
  - [ ] 3.1.4 Load audio and extract features using `EnergyDetector._extract_features()`
  - [ ] 3.1.5 Calculate per-segment average features
  - [ ] 3.1.6 Use multi-feature scoring to assign EDM labels

- [ ] 3.2 Implement `HybridDetector._score_segment()` with drop/breakdown scoring logic

- [ ] 3.3 Add `HybridDetector` to `get_detector()` function with `detector_type="hybrid"`

## 4. API Integration

- [ ] 4.1 Update `analyze_structure()` in `src/edm/analysis/structure.py`
  - [ ] 4.1.1 Add `hybrid` option to detector parameter docstring
  - [ ] 4.1.2 Update example usage in docstring

- [ ] 4.2 Update `get_detector()` in `src/edm/analysis/structure_detector.py`
  - [ ] 4.2.1 Add `elif detector_type == "hybrid": return HybridDetector()`

## 5. Testing

- [ ] 5.1 Unit tests for feature extraction
  - [ ] 5.1.1 Test `_extract_features()` output shapes match expected frame count
  - [ ] 5.1.2 Test `_normalize()` handles zero variance arrays
  - [ ] 5.1.3 Test feature values are in [0, 1] range

- [ ] 5.2 Unit tests for scoring
  - [ ] 5.2.1 Test `_score_segment()` returns valid labels from EDM vocabulary
  - [ ] 5.2.2 Test confidence scores are in [0, 1] range
  - [ ] 5.2.3 Test drop scoring with high bass/energy profile
  - [ ] 5.2.4 Test breakdown scoring with low energy profile

- [ ] 5.3 Integration tests
  - [ ] 5.3.1 Test `EnergyDetector` runs without errors on sample audio
  - [ ] 5.3.2 Test `HybridDetector` runs without errors on sample audio
  - [ ] 5.3.3 Test sections have no overlaps and full track coverage
  - [ ] 5.3.4 Test performance: analysis completes in <30s per track

## 6. Evaluation

- [ ] 6.1 Run baseline evaluation (current RMS-only detector)
  - [ ] 6.1.1 Analyze all annotated tracks with `detector=energy` (old)
  - [ ] 6.1.2 Run `/evaluate` and record metrics

- [ ] 6.2 Run multi-feature evaluation
  - [ ] 6.2.1 Analyze all annotated tracks with `detector=energy` (new multi-feature)
  - [ ] 6.2.2 Run `/evaluate` and compare to baseline

- [ ] 6.3 Run hybrid evaluation
  - [ ] 6.3.1 Analyze all annotated tracks with `detector=hybrid`
  - [ ] 6.3.2 Run `/evaluate` and compare to baseline and multi-feature

- [ ] 6.4 Document results
  - [ ] 6.4.1 Create comparison table: boundary F1, drop precision/recall, analysis time
  - [ ] 6.4.2 Identify which approach performs best

## 7. Optimization (If Needed)

- [ ] 7.1 Profile feature extraction time
  - [ ] 7.1.1 Identify bottlenecks (STFT computation, normalization, etc.)
  - [ ] 7.1.2 Cache STFT if computed multiple times

- [ ] 7.2 Tune feature weights
  - [ ] 7.2.1 Grid search or manual tuning of drop_score weights
  - [ ] 7.2.2 Grid search or manual tuning of breakdown_score weights
  - [ ] 7.2.3 Re-evaluate after tuning

- [ ] 7.3 Adjust thresholds
  - [ ] 7.3.1 Experiment with drop_score threshold (0.60-0.70)
  - [ ] 7.3.2 Experiment with breakdown_score threshold (0.55-0.65)

## 8. Documentation

- [ ] 8.1 Update analysis spec (`openspec/specs/analysis/spec.md`)
  - [ ] 8.1.1 Add requirement for multi-feature energy detection
  - [ ] 8.1.2 Document supported features (spectral centroid, contrast, band energy, onset strength)
  - [ ] 8.1.3 Add hybrid detector scenarios

- [ ] 8.2 Update docstrings
  - [ ] 8.2.1 Document `_extract_features()` method
  - [ ] 8.2.2 Document `HybridDetector` class
  - [ ] 8.2.3 Update `analyze_structure()` docstring with hybrid option

- [ ] 8.3 Add code comments explaining feature weights and thresholds

## 9. Archive (When Complete)

- [ ] 9.1 Update `openspec/specs/analysis/spec.md` with finalized multi-feature energy requirements
- [ ] 9.2 Run `openspec archive ENERGY-multi-feature-energy-detection --yes`
- [ ] 9.3 Verify change moved to `openspec/changes/archive/`
