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

## 4. Downbeat/Structure Validator

- [ ] 4.1 Implement `DownbeatStructureValidator` class (`downbeat_structure.py`)
- [ ] 4.2 Implement `_find_nearest_downbeat()` - find closest downbeat to boundary
- [ ] 4.3 Implement `_calculate_downbeat_alignment()` - measure boundary-downbeat offset
- [ ] 4.4 Implement `_validate_first_downbeat()` - check first downbeat vs structure start
- [ ] 4.5 Add unit tests for downbeat alignment validation

## 5. Beat Grid Validator

- [ ] 5.1 Implement `BeatGridValidator` class (`beat_grid.py`)
- [ ] 5.2 Implement `_validate_beat_bpm_consistency()` - check beat intervals match BPM
- [ ] 5.3 Implement `_validate_downbeat_spacing()` - check downbeats at 4-beat intervals
- [ ] 5.4 Implement `_detect_missing_beats()` - find gaps in beat sequence
- [ ] 5.5 Add unit tests for beat grid validation

## 6. Confidence-Weighted Arbitration

- [ ] 6.1 Implement `_arbitrate_conflict()` - compare confidence scores
- [ ] 6.2 Implement suggestion generation based on confidence winner
- [ ] 6.3 Implement `needs_review` flagging for ambiguous conflicts
- [ ] 6.4 Add tests for arbitration logic

## 7. Orchestration

- [ ] 7.1 Implement `ValidationOrchestrator` class (`orchestrator.py`)
- [ ] 7.2 Implement `validate()` method - run all applicable validators
- [ ] 7.3 Implement `should_validate()` - auto-trigger based on confidence
- [ ] 7.4 Add integration tests for orchestrator

## 8. CLI Integration

- [ ] 8.1 Add `--validate` / `--no-validate` flag to analyze command
- [ ] 8.2 Integrate validation into analyze flow (post-analysis)
- [ ] 8.3 Add validation section to JSON output (including uncertainty)
- [ ] 8.4 Add end-to-end tests

## 9. Documentation

- [ ] 9.1 Update CLI reference docs
- [ ] 9.2 Add validation section to architecture docs
