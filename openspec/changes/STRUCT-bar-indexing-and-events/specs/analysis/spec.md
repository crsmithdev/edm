# analysis Specification Delta

## ADDED Requirements

### Requirement: 1-Indexed Bar Numbers

The system SHALL use 1-indexed bar numbering in all structure output to match DJ software conventions (Rekordbox, Traktor, Ableton, Serato).

#### Scenario: First section starts at bar 1
- **WHEN** structure analysis is performed on a track
- **THEN** the first section's start_bar is 1 (not 0)

#### Scenario: Bar span calculation
- **WHEN** an 8-bar intro is detected from time 0.0 to 15.0 seconds at 128 BPM
- **THEN** it is output as `[1, 8, intro]` representing bars 1 through 8 inclusive

#### Scenario: Consecutive sections don't overlap
- **WHEN** structure contains sections `[1, 8, intro]` and `[9, 16, buildup]`
- **THEN** bar 9 immediately follows bar 8 with no gap or overlap

### Requirement: Event vs Span Separation

The system SHALL distinguish between moment-based events (drops, percussion onsets) and time-spanning sections (intro, breakdown, buildup).

#### Scenario: Drop as event
- **WHEN** a drop is detected at bar 33
- **THEN** it appears in the `events` list as `[33, drop]`, not as a span

#### Scenario: Percussion onset as event
- **WHEN** a strong kick onset is detected at bar 17
- **THEN** it appears in the `events` list as `[17, kick]`

#### Scenario: Buildup as span
- **WHEN** a buildup is detected from bars 25-32
- **THEN** it appears in the `structure` list as `[25, 32, buildup]`

#### Scenario: Output contains both structures and events
- **WHEN** structure analysis completes
- **THEN** the StructureResult contains both `structure` (list of spans) and `events` (list of moments)

### Requirement: Merge Consecutive 'other' Sections

The system SHALL merge consecutive sections labeled 'other' into a single span for clarity.

#### Scenario: Merge two consecutive 'other' sections
- **WHEN** detection produces sections `[1, 16, other]` and `[16, 24, other]`
- **THEN** they are merged into a single section `[1, 24, other]`

#### Scenario: Non-consecutive 'other' sections remain separate
- **WHEN** detection produces `[1, 16, other]`, `[16, 24, drop]`, `[24, 32, other]`
- **THEN** the two 'other' sections remain separate (not merged)

#### Scenario: Merged confidence is maximum
- **WHEN** merging sections with confidence 0.6 and 0.8
- **THEN** the merged section has confidence 0.8

### Requirement: Event Detection

The system SHALL detect moment-based events including drops and percussion onsets.

#### Scenario: Drop detection
- **WHEN** a section is labeled 'drop' by the energy detector
- **THEN** it is marked as an event (`is_event=True`) and output with only its start bar

#### Scenario: Kick detection
- **WHEN** a strong percussion onset is detected via librosa onset detection
- **THEN** it is added to the events list as `[bar, kick]`

#### Scenario: Kick filtering
- **WHEN** multiple weak onsets are detected
- **THEN** only onsets exceeding a strength threshold are output as kick events

### Requirement: Raw Detection Output

The system SHALL include raw detection data in structure output for debugging and analysis improvement.

#### Scenario: Raw output contains all detected sections
- **WHEN** structure analysis completes
- **THEN** the output includes a `raw` key with all original detected sections before post-processing

#### Scenario: Raw sections include timestamps
- **WHEN** a raw section is output
- **THEN** it includes `start` and `end` times in seconds

#### Scenario: Raw sections include fractional bars
- **WHEN** a raw section is output and BPM is available
- **THEN** it includes `start_bar` and `end_bar` as fractional values (e.g., 24.1, 32.3)

#### Scenario: Raw sections include confidence
- **WHEN** a raw section is output
- **THEN** it includes a `confidence` score between 0 and 1

#### Scenario: Raw sections include label
- **WHEN** a raw section is output
- **THEN** it includes the original `label` assigned by the detector

### Requirement: Annotation Template Output

The system SHALL output a simplified annotation template when requested for manual ground truth creation.

#### Scenario: Annotations flag generates template file
- **WHEN** analyze is run with `--annotations` flag
- **THEN** a `.annotations.yaml` file is created alongside standard output

#### Scenario: Template contains metadata
- **WHEN** an annotation template is generated
- **THEN** it includes file, duration, bpm, downbeat, and time_signature

#### Scenario: Template uses simple bar/label format
- **WHEN** an annotation template is generated
- **THEN** the `annotations` key contains a list of `[bar, label]` tuples

#### Scenario: Template derives from detection
- **WHEN** an annotation template is generated
- **THEN** initial annotations are derived from detected sections/events as a starting point for editing

