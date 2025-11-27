## MODIFIED Requirements

### Requirement: Audio Analysis Module

The library SHALL provide an analysis module (`edm.analysis`) that contains functions for analyzing EDM tracks using beat_this for neural network-based BPM detection and librosa as fallback.

#### Scenario: Analyze track for BPM
- **WHEN** user calls `analyze_track(filepath, config)` with BPM detection enabled
- **THEN** returns a result object containing BPM value and confidence score

#### Scenario: Analyze track structure
- **WHEN** user calls `analyze_track(filepath, config)` with structure detection enabled
- **THEN** returns a result object containing detected sections (intro, drop, breakdown, etc.)

#### Scenario: Handle invalid audio file
- **WHEN** user calls `analyze_track(filepath, config)` with an invalid file path
- **THEN** raises a custom `AudioFileError` exception with descriptive message

#### Scenario: BPM computation method selection
- **WHEN** user calls `compute_bpm(filepath, prefer_madmom=True)`
- **THEN** uses beat_this (neural network approach) as primary detector with librosa fallback

#### Scenario: BPM computation with librosa preference
- **WHEN** user calls `compute_bpm(filepath, prefer_madmom=False)`
- **THEN** uses librosa (traditional DSP approach) as primary detector
