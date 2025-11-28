## ADDED Requirements

### Requirement: Cross-Validation Framework

The system SHALL provide a pluggable cross-validation framework that detects inconsistencies between analysis results.

#### Scenario: Validator registration
- **WHEN** a validator is added to the orchestrator
- **THEN** it is called during validation if applicable to the analysis results

#### Scenario: Multiple validators
- **WHEN** multiple validators are registered
- **THEN** all applicable validators run and results are aggregated

#### Scenario: Validator applicability
- **WHEN** a validator's `is_applicable()` returns False
- **THEN** that validator is skipped without error

### Requirement: Beat/Structure Alignment Validation

The system SHALL validate that structure boundaries align with bar positions based on detected BPM.

#### Scenario: Calculate alignment for each boundary
- **WHEN** validation runs on analysis results
- **THEN** each structure boundary's bar position is calculated and offset from nearest whole bar is recorded

#### Scenario: Detect systematic BPM offset
- **WHEN** all boundaries are consistently offset by similar fraction (low std dev)
- **THEN** error pattern is classified as `BPM_SYSTEMATIC_OFFSET`

#### Scenario: Detect downbeat phase error
- **WHEN** boundaries are consistently offset by approximately 0.5 bars
- **THEN** error pattern is classified as `DOWNBEAT_PHASE_ERROR`

#### Scenario: Detect single boundary error
- **WHEN** one boundary is misaligned but others are aligned
- **THEN** error pattern is classified as `SINGLE_BOUNDARY_ERROR`

#### Scenario: Detect progressive drift
- **WHEN** alignment error increases progressively through the track
- **THEN** error pattern is classified as `PROGRESSIVE_DRIFT` and flagged for manual review

#### Scenario: Pass validation when aligned
- **WHEN** all boundaries are within tolerance (default: 0.25 beats / 1/16 bar)
- **THEN** validation passes with error pattern `NONE`

### Requirement: Validation Result Reporting

The system SHALL report validation results with error patterns, alignment metrics, confidence scores, and uncertainty measures.

#### Scenario: Report alignment ratio
- **WHEN** validation completes
- **THEN** result includes ratio of aligned boundaries to total boundaries

#### Scenario: Report error pattern
- **WHEN** validation detects a pattern
- **THEN** result includes pattern classification and confidence score

#### Scenario: Report per-boundary errors
- **WHEN** validation finds misaligned boundaries
- **THEN** result includes list of boundaries with their offset values

#### Scenario: Report uncertainty separately from confidence
- **WHEN** validation detects disagreement between signals
- **THEN** result includes both `confidence` (detector certainty) and `uncertainty` (signal disagreement)

#### Scenario: Flag high-confidence conflicts
- **WHEN** both signals have high confidence but disagree significantly
- **THEN** result flags as `needs_review` with explanation

### Requirement: Confidence-Weighted Arbitration

The system SHALL use confidence scores to weight which signal to trust when signals conflict.

#### Scenario: Trust higher-confidence signal
- **WHEN** BPM confidence exceeds structure confidence by >0.2
- **THEN** validation suggests quantizing structure boundaries to BPM grid

#### Scenario: Suggest BPM correction
- **WHEN** structure confidence exceeds BPM confidence by >0.2
- **THEN** validation suggests BPM correction based on structure alignment

#### Scenario: Flag ambiguous conflicts
- **WHEN** confidence scores are within 0.2 of each other and signals disagree
- **THEN** validation flags for manual review without suggesting correction

### Requirement: Fault Detection Before Fusion

The system SHALL validate each signal source independently before cross-validation.

#### Scenario: Skip invalid BPM
- **WHEN** BPM is outside valid range (40-200) or confidence < 0.3
- **THEN** cross-validation is skipped with warning

#### Scenario: Skip invalid structure
- **WHEN** fewer than 2 sections detected or all sections have confidence < 0.3
- **THEN** cross-validation is skipped with warning

#### Scenario: Proceed when both valid
- **WHEN** both BPM and structure pass basic validity checks
- **THEN** cross-validation proceeds normally

### Requirement: CLI Validation Integration

The system SHALL support validation through the CLI analyze command.

#### Scenario: Explicit validation flag
- **WHEN** `edm analyze --validate` is run
- **THEN** validation runs regardless of analysis confidence

#### Scenario: Disable validation
- **WHEN** `edm analyze --no-validate` is run
- **THEN** validation is skipped

#### Scenario: Auto-trigger validation
- **WHEN** `edm analyze` is run without flag and analysis confidence < 0.8
- **THEN** validation runs automatically

#### Scenario: Validation output in JSON
- **WHEN** validation runs with JSON output
- **THEN** result includes `validation` object with `passed`, `confidence`, `error_pattern`, and `details`
