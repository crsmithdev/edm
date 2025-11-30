## MODIFIED Requirements

### Requirement: Structure Analysis
The system SHALL analyze audio files to identify structural elements including sections (spans) and events (moments).

#### Scenario: Detect drop events
- **WHEN** analyzing a track with energy transitions
- **THEN** drop moments are identified as events `[bar, "drop"]`

#### Scenario: Detect section spans
- **WHEN** analyzing a track structure
- **THEN** sections are identified as spans `[start_bar, end_bar, label]`

#### Scenario: Polymorphic output
- **WHEN** structure analysis completes
- **THEN** output contains mixed 2-element (event) and 3-element (span) tuples

#### Scenario: Event labels
- **WHEN** a structural moment is detected
- **THEN** label is one of: `drop`

#### Scenario: Span labels
- **WHEN** a structural section is detected
- **THEN** label is one of: `intro`, `outro`, `breakdown`, `buildup`, `verse`, `chorus`, `bridge`, `other`
