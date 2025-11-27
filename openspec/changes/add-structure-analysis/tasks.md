# Tasks: Add Structure Analysis

## 1. Core Infrastructure

- [ ] 1.1 Add `allin1` dependency to pyproject.toml
- [ ] 1.2 Create `src/edm/analysis/structure_detector.py` with detector protocol
- [ ] 1.3 Add structure detection configuration to `src/edm/config.py`
- [ ] 1.4 Update `src/edm/analysis/__init__.py` exports

## 2. Allin1 Detector Implementation

- [ ] 2.1 Implement `Allin1Detector` class with model loading
- [ ] 2.2 Add GPU/CPU device selection logic
- [ ] 2.3 Implement section label mapping (chorus→drop, bridge→breakdown, etc.)
- [ ] 2.4 Add confidence score extraction from model output
- [ ] 2.5 Handle model import failure gracefully (return None, log warning)

## 3. Energy-Based Detector Implementation

- [ ] 3.1 Implement `EnergyDetector` class using librosa
- [ ] 3.2 Add RMS energy analysis for drop detection
- [ ] 3.3 Add spectral contrast analysis for bass-heavy sections
- [ ] 3.4 Implement section boundary refinement (minimum duration filtering)
- [ ] 3.5 Add intro/outro detection based on energy envelope

## 4. Main Analysis Function

- [ ] 4.1 Refactor `analyze_structure()` to use detector pattern
- [ ] 4.2 Implement detector selection logic (auto/allin1/energy)
- [ ] 4.3 Add section post-processing (gap filling, overlap resolution)
- [ ] 4.4 Ensure full track coverage (sections span 0 to duration)
- [ ] 4.5 Integrate with audio caching for efficiency

## 5. CLI Integration

- [ ] 5.1 Add `--structure-detector` option to analyze command
- [ ] 5.2 Update `--types structure` to use new implementation
- [ ] 5.3 Format structure output in CLI results display
- [ ] 5.4 Add structure sections to JSON output format

## 6. Evaluation Framework

- [ ] 6.1 Create `src/edm/evaluation/structure.py` evaluator
- [ ] 6.2 Define ground truth CSV format for structure annotations
- [ ] 6.3 Implement boundary tolerance matching algorithm
- [ ] 6.4 Calculate per-section-type precision/recall/F1
- [ ] 6.5 Add `edm evaluate structure` subcommand

## 7. Testing

- [ ] 7.1 Unit tests for Allin1Detector (mocked model)
- [ ] 7.2 Unit tests for EnergyDetector
- [ ] 7.3 Unit tests for section post-processing
- [ ] 7.4 Integration test with real audio file
- [ ] 7.5 Test detector fallback behavior

## 8. Documentation

- [ ] 8.1 Update docs/cli-reference.md with structure options
- [ ] 8.2 Update docs/architecture.md with structure detection design
- [ ] 8.3 Add structure ground truth format documentation
- [ ] 8.4 Document accuracy targets and evaluation methodology

## Dependencies

- Tasks 2.x depend on 1.x (infrastructure first)
- Tasks 3.x can run parallel to 2.x (independent detector)
- Task 4.x depends on both 2.x and 3.x
- Task 5.x depends on 4.x
- Task 6.x can start after 4.1-4.2
- Task 7.x runs throughout
- Task 8.x runs after implementation complete

## Validation Criteria

- [ ] `openspec validate add-structure-analysis --strict` passes
- [ ] All tests pass: `uv run pytest tests/test_analysis/`
- [ ] Structure analysis returns real sections (not hardcoded)
- [ ] Drop detection precision >90% on test set
- [ ] Processing time <30 seconds per track
