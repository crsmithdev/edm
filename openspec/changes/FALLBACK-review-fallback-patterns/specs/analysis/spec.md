## MODIFIED Requirements

### Requirement: Structure Detection

The system SHALL detect track structure sections (intro, buildup, drop, breakdown, outro) using MSAF-based analysis.

#### Scenario: Analyze track structure with MSAF
- **WHEN** `analyze_structure(filepath)` is called
- **THEN** returns StructureResult containing detected sections with labels, timestamps, and confidence scores

#### Scenario: Explicit detector selection
- **WHEN** `analyze_structure(filepath, detector="energy")` is called
- **THEN** uses the specified detector
- **WHEN** `analyze_structure(filepath, detector="msaf")` is called
- **THEN** uses MSAF detector

#### Scenario: MSAF failure
- **WHEN** MSAF detection fails for any reason
- **THEN** raises AnalysisError with descriptive message
- **AND** does not silently fall back to another detector

### Requirement: Energy-Based Detection

The system SHALL provide rule-based structure detection using audio energy analysis as an explicit alternative detector (not automatic fallback).

#### Scenario: Drop detection via RMS energy
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies high-energy sustained sections as potential drops

#### Scenario: Breakdown detection via energy dip
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies low-energy sections following drops as breakdowns

#### Scenario: Buildup detection via energy rise
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies sections with rising energy before drops as buildups

#### Scenario: Section merging
- **WHEN** energy-based detector identifies sections
- **THEN** sections shorter than 4 bars (at detected BPM) are merged with adjacent sections

## REMOVED Requirements

### Requirement: Automatic MSAF Fallback

#### Scenario: Analyze track structure with energy fallback
- **WHEN** `analyze_structure(filepath)` is called and MSAF is unavailable
- **THEN** falls back to energy-based detection and returns StructureResult with detected sections
