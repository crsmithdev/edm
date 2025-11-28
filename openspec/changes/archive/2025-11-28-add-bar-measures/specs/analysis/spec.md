## ADDED Requirements

### Requirement: Bar/Measure Calculation

The system SHALL calculate bar (measure) positions from BPM and time signatures to enable musical structure representation.

#### Scenario: Time to bars conversion
- **WHEN** converting a time position (e.g., 30.5 seconds) with known BPM (128) and time signature (4/4)
- **THEN** return the bar number and fractional beat position (e.g., bar 16, beat 1.0)

#### Scenario: Default time signature assumption
- **WHEN** no time signature is provided
- **THEN** assume 4/4 time (4 beats per bar) as standard for EDM

#### Scenario: Bar count for time range
- **WHEN** calculating bars for a section (e.g., 30.5s to 61.0s at 128 BPM)
- **THEN** return the number of bars spanning that range (e.g., 16 bars)

#### Scenario: Handling BPM unavailable
- **WHEN** BPM is not available or detection failed
- **THEN** bar calculations return None and time-based values are still usable

### Requirement: Bar-Annotated Structure Sections

Structure analysis results SHALL include bar counts and positions for each detected section.

#### Scenario: Section with bar metadata
- **WHEN** structure analysis detects a section with start=30.5s, end=61.0s at 128 BPM
- **THEN** result includes start_bar, end_bar, and bar_count fields (e.g., bar 16 to bar 32, 16 bars)

#### Scenario: Structure output without BPM
- **WHEN** structure analysis runs but BPM detection failed
- **THEN** bar fields are None and section times remain valid

#### Scenario: CLI display of bar counts
- **WHEN** displaying structure results in CLI table format
- **THEN** show bar counts alongside time (e.g., "16 bars (30.5s-61.0s)")

### Requirement: Beat Grid Foundation

Bar/measure utilities SHALL be designed to integrate with future beat grid implementation.

#### Scenario: Time-to-bar with beat grid
- **WHEN** a beat grid is available (future implementation)
- **THEN** bar calculations use actual beat positions instead of constant tempo assumption

#### Scenario: Bar utilities without beat grid
- **WHEN** beat grid is not yet implemented
- **THEN** bar calculations use BPM and time signature with constant tempo assumption

#### Scenario: Extensible bar calculation interface
- **WHEN** implementing bar calculation utilities
- **THEN** design accepts optional beat grid parameter for future compatibility

### Requirement: Bar-Based Section Queries

The system SHALL support querying structure sections by bar position.

#### Scenario: Get section at bar position
- **WHEN** querying "what section is at bar 64?"
- **THEN** return the section containing that bar position (e.g., "drop")

#### Scenario: Bar position outside track range
- **WHEN** querying a bar position beyond track duration
- **THEN** return None or raise appropriate error

#### Scenario: Section boundary alignment check
- **WHEN** analyzing section boundaries
- **THEN** report whether boundaries align with bar boundaries (within tolerance)

### Requirement: Bar-Based Ground Truth Annotations

Ground truth annotation files SHALL support bar-based positions for section boundaries to avoid time conversion friction.

#### Scenario: CSV annotations with bar positions
- **WHEN** loading structure annotations from CSV
- **THEN** accept columns `start_bar` and `end_bar` instead of `start` and `end` time values

#### Scenario: Bar to time conversion for evaluation
- **WHEN** evaluating against bar-based annotations
- **THEN** convert bar positions to time using track BPM before comparing with detected sections

#### Scenario: Mixed annotation formats
- **WHEN** annotation file contains both time and bar columns
- **THEN** prefer bar-based positions if BPM is available, fall back to time-based

#### Scenario: Natural annotation workflow
- **WHEN** annotating structure while listening to a track
- **THEN** annotator can specify "drop at bar 64" instead of converting player display (e.g., "2:43") to seconds

#### Scenario: BPM metadata for bar annotations
- **WHEN** using bar-based annotations
- **THEN** CSV includes BPM column or separate BPM metadata file to enable time conversion
