# Tasks: Add Structure Analysis

## 1. Core Infrastructure

- [x] 1.1 Add `msaf` dependency to pyproject.toml
- [x] 1.2 Create `src/edm/analysis/structure_detector.py` with detector protocol
- [x] 1.3 Add structure detection configuration to `src/edm/config.py`
- [x] 1.4 Update `src/edm/analysis/__init__.py` exports

## 2. MSAF Detector Implementation

- [x] 2.1 Implement `MSAFDetector` class with boundary detection
- [x] 2.2 Add segment labeling using MSAF algorithms
- [x] 2.3 Implement EDM label mapping based on energy characteristics
- [x] 2.4 Add confidence score calculation from boundary strength
- [x] 2.5 Handle msaf import failure gracefully (return None, log warning)

## 3. Energy-Based Detector Implementation

- [x] 3.1 Implement `EnergyDetector` class using librosa
- [x] 3.2 Add RMS energy analysis for drop detection
- [x] 3.3 Add spectral contrast analysis for bass-heavy sections
- [x] 3.4 Implement section boundary refinement (minimum duration filtering)
- [x] 3.5 Add intro/outro detection based on energy envelope

## 4. Main Analysis Function

- [x] 4.1 Refactor `analyze_structure()` to use detector pattern
- [x] 4.2 Implement detector selection logic (auto/msaf/energy)
- [x] 4.3 Add section post-processing (gap filling, overlap resolution)
- [x] 4.4 Ensure full track coverage (sections span 0 to duration)
- [x] 4.5 Integrate with audio caching for efficiency

## 5. CLI Integration

- [x] 5.1 Add `--structure-detector` option to analyze command
- [x] 5.2 Update `--types structure` to use new implementation
- [x] 5.3 Format structure output in CLI results display
- [x] 5.4 Add structure sections to JSON output format

## 6. Evaluation Framework

- [x] 6.1 Create `src/edm/evaluation/evaluators/structure.py` evaluator
- [x] 6.2 Define ground truth CSV format for structure annotations
- [x] 6.3 Implement boundary tolerance matching algorithm
- [x] 6.4 Calculate per-section-type precision/recall/F1
- [x] 6.5 Add `edm evaluate structure` subcommand

## 7. Testing

- [x] 7.1 Unit tests for MSAFDetector (mocked)
- [x] 7.2 Unit tests for EnergyDetector
- [x] 7.3 Unit tests for section post-processing
- [x] 7.4 Integration test with real audio file
- [x] 7.5 Test detector fallback behavior

## 8. Documentation

- [x] 8.1 Update docs/cli-reference.md with structure options
- [x] 8.2 Update docs/architecture.md with structure detection design
- [x] 8.3 Add structure ground truth format documentation
- [x] 8.4 Document accuracy targets and evaluation methodology

## Dependencies

- Tasks 2.x depend on 1.x (infrastructure first)
- Tasks 3.x can run parallel to 2.x (independent detector)
- Task 4.x depends on both 2.x and 3.x
- Task 5.x depends on 4.x
- Task 6.x can start after 4.1-4.2
- Task 7.x runs throughout
- Task 8.x runs after implementation complete

## Validation Criteria

- [x] `openspec validate add-structure-analysis --strict` passes
- [x] All tests pass: `uv run pytest tests/`
- [x] Structure analysis returns real sections (not hardcoded)
- [ ] Drop detection precision >90% on test set (requires ground truth data - pending user annotations)
- [x] Processing time acceptable: energy detector ~2-8s/track, msaf ~20-40s/track, parallel processing with 3.2x speedup verified
