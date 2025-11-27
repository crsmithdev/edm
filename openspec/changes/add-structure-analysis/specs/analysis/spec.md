## MODIFIED Requirements

### Requirement: Audio Caching

The system SHALL cache loaded audio data to avoid redundant file I/O and decoding operations during analysis.

#### Scenario: Cache hit on repeated load
- **WHEN** the same audio file is loaded multiple times within a session
- **THEN** the audio data is decoded only once and subsequent loads return cached data

#### Scenario: Cache eviction on size limit
- **WHEN** the cache reaches its configured size limit
- **THEN** the least recently used audio data is evicted to make room for new entries

#### Scenario: Cache bypass for memory-constrained environments
- **WHEN** cache size is set to 0
- **THEN** audio caching is disabled and each load performs fresh decoding

#### Scenario: Structure analysis with cached audio
- **WHEN** structure analysis is performed after BPM analysis on the same file
- **THEN** the cached audio data is reused without reloading from disk

## ADDED Requirements

### Requirement: Structure Detection

The system SHALL detect track structure sections (intro, buildup, drop, breakdown, outro) using ML-based analysis with rule-based fallback.

#### Scenario: Analyze track structure with Allin1
- **WHEN** `analyze_structure(filepath)` is called with Allin1 available
- **THEN** returns StructureResult containing detected sections with labels, timestamps, and confidence scores

#### Scenario: Analyze track structure with energy fallback
- **WHEN** `analyze_structure(filepath)` is called and Allin1 is unavailable
- **THEN** falls back to energy-based detection and returns StructureResult with detected sections

#### Scenario: Explicit detector selection
- **WHEN** `analyze_structure(filepath, detector="energy")` is called
- **THEN** uses the specified detector regardless of Allin1 availability

#### Scenario: Auto detector selection
- **WHEN** `analyze_structure(filepath, detector="auto")` is called (default)
- **THEN** uses Allin1 if available, otherwise falls back to energy-based detection

#### Scenario: GPU acceleration
- **WHEN** CUDA is available and Allin1 detector is used
- **THEN** inference runs on GPU for faster processing

#### Scenario: CPU fallback
- **WHEN** CUDA is not available and Allin1 detector is used
- **THEN** inference runs on CPU without errors

### Requirement: Section Data Model

The system SHALL represent detected sections with standardized labels, timestamps, and confidence scores.

#### Scenario: Section contains required fields
- **WHEN** a Section object is created
- **THEN** it contains label (str), start_time (float), end_time (float), and confidence (float)

#### Scenario: EDM-specific section labels
- **WHEN** structure analysis completes
- **THEN** sections use EDM terminology: intro, buildup, drop, breakdown, outro

#### Scenario: Chronological ordering
- **WHEN** StructureResult is returned
- **THEN** sections are ordered by start_time ascending with no overlaps

#### Scenario: Full track coverage
- **WHEN** structure analysis completes
- **THEN** sections cover the entire track duration from 0.0 to track end with no gaps

### Requirement: Drop Detection Accuracy

The system SHALL detect drop sections with precision >90% and recall >85% against ground truth.

#### Scenario: Drop precision validation
- **WHEN** structure analysis is evaluated against ground truth annotations
- **THEN** drop detection precision exceeds 90% (low false positive rate)

#### Scenario: Drop recall validation
- **WHEN** structure analysis is evaluated against ground truth annotations
- **THEN** drop detection recall exceeds 85% (few missed drops)

#### Scenario: Boundary tolerance
- **WHEN** comparing detected boundaries to ground truth
- **THEN** boundaries within ±2 seconds of ground truth are considered correct

### Requirement: Energy-Based Detection

The system SHALL provide rule-based structure detection using audio energy analysis as a fallback method.

#### Scenario: Drop detection via RMS energy
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies high-energy sustained sections as potential drops

#### Scenario: Breakdown detection via energy dip
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies low-energy sections following drops as breakdowns

#### Scenario: Buildup detection via energy rise
- **WHEN** energy-based detector analyzes a track
- **THEN** identifies sections with rising energy before drops as buildups

#### Scenario: Minimum section duration
- **WHEN** energy-based detector identifies sections
- **THEN** sections shorter than 4 bars (at detected BPM) are merged with adjacent sections

### Requirement: Structure Evaluation

The system SHALL support accuracy evaluation of structure detection against ground truth annotations.

#### Scenario: Evaluate structure accuracy
- **WHEN** `edm evaluate structure --source <dir> --reference <csv>` is run
- **THEN** outputs precision, recall, and F1 scores for each section type

#### Scenario: Boundary tolerance configuration
- **WHEN** `--tolerance 3.0` flag is provided to evaluation
- **THEN** boundaries within ±3 seconds are considered correct matches

#### Scenario: Per-track results
- **WHEN** evaluation completes
- **THEN** outputs per-track results showing detected vs expected sections

#### Scenario: Missing ground truth handling
- **WHEN** a track lacks ground truth annotations
- **THEN** track is skipped with warning and excluded from aggregate metrics
