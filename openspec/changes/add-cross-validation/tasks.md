# Tasks

## 1. Foundation

- [ ] 1.1 Create `src/edm/analysis/validation/` module directory
- [ ] 1.2 Implement result dataclasses (`results.py`): `ErrorPattern`, `AlignmentError`, `CorrectionProposal`, `ValidationResult`, `CrossValidationResult`
- [ ] 1.3 Add `uncertainty` field to results (separate from `confidence`)
- [ ] 1.4 Implement validator protocol and base class (`base.py`)
- [ ] 1.5 Add unit tests for result dataclasses

## 2. Fault Detection (FDE)

- [ ] 2.1 Implement `_validate_bpm_source()` - check BPM in range, confidence > 0.3
- [ ] 2.2 Implement `_validate_structure_source()` - check >= 2 sections, confidence > 0.3
- [ ] 2.3 Add tests for source validation edge cases

## 3. Beat/Structure Validator

- [ ] 3.1 Implement `BeatStructureValidator` class (`beat_structure.py`)
- [ ] 3.2 Implement `_calculate_alignment_errors()` - compute bar offset for each boundary
- [ ] 3.3 Implement `_detect_error_pattern()` - classify error type from offset distribution
- [ ] 3.4 Implement `_calculate_uncertainty()` - measure signal disagreement
- [ ] 3.5 Add unit tests with synthetic alignment data

## 4. Confidence-Weighted Arbitration

- [ ] 4.1 Implement `_arbitrate_conflict()` - compare confidence scores
- [ ] 4.2 Implement suggestion generation based on confidence winner
- [ ] 4.3 Implement `needs_review` flagging for ambiguous conflicts
- [ ] 4.4 Add tests for arbitration logic

## 5. Orchestration

- [ ] 5.1 Implement `ValidationOrchestrator` class (`orchestrator.py`)
- [ ] 5.2 Implement `validate()` method - run all applicable validators
- [ ] 5.3 Implement `should_validate()` - auto-trigger based on confidence
- [ ] 5.4 Add integration tests for orchestrator

## 6. CLI Integration

- [ ] 6.1 Add `--validate` / `--no-validate` flag to analyze command
- [ ] 6.2 Integrate validation into analyze flow (post-analysis)
- [ ] 6.3 Add validation section to JSON output (including uncertainty)
- [ ] 6.4 Add end-to-end tests

## 7. Documentation

- [ ] 7.1 Update CLI reference docs
- [ ] 7.2 Add validation section to architecture docs
