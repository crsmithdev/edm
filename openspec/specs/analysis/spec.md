# analysis Specification

## Purpose
TBD - created by archiving change add-audio-caching. Update Purpose after archive.
## Requirements
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

### Requirement: Pre-loaded Audio Support

The BPM detection functions SHALL accept pre-loaded audio data to enable cache integration.

#### Scenario: BPM detection with pre-loaded audio
- **WHEN** `compute_bpm_librosa()` is called with `audio` parameter containing `(y, sr)` tuple
- **THEN** the function uses the provided audio data instead of loading from disk

#### Scenario: BPM detection with filepath only
- **WHEN** `compute_bpm_librosa()` is called without `audio` parameter
- **THEN** the function loads audio from the filepath (backward compatible)

#### Scenario: beat_this detection with filepath
- **WHEN** `compute_bpm_beat_this()` is called with a filepath
- **THEN** the function loads audio internally and returns BPM calculated from beat intervals

### Requirement: beat_this BPM Detection

The system SHALL use beat_this as the primary neural network-based BPM detector, replacing madmom.

#### Scenario: BPM detection with beat_this
- **WHEN** `compute_bpm_beat_this()` is called with a valid audio filepath
- **THEN** returns a `ComputedBPM` result with BPM calculated from beat intervals, confidence based on interval consistency, and method set to "beat-this"

#### Scenario: CUDA device selection
- **WHEN** CUDA is available on the system
- **THEN** beat_this uses GPU acceleration for faster inference

#### Scenario: CPU fallback
- **WHEN** CUDA is not available
- **THEN** beat_this falls back to CPU inference without errors

#### Scenario: beat_this import failure fallback
- **WHEN** beat_this is not installed or import fails
- **THEN** `compute_bpm()` falls back to librosa for BPM detection

#### Scenario: EDM range adjustment
- **WHEN** beat_this detects a BPM outside the preferred EDM range (120-150)
- **THEN** alternatives at half/double tempo are considered and the value in range is preferred

### Requirement: Structure Detection

The system SHALL detect track structure sections (intro, buildup, drop, breakdown, outro) using MSAF-based analysis with energy-based fallback.

#### Scenario: Analyze track structure with MSAF
- **WHEN** `analyze_structure(filepath)` is called with MSAF available
- **THEN** returns StructureResult containing detected sections with labels, timestamps, and confidence scores

#### Scenario: Analyze track structure with energy fallback
- **WHEN** `analyze_structure(filepath)` is called and MSAF is unavailable
- **THEN** falls back to energy-based detection and returns StructureResult with detected sections

#### Scenario: Explicit detector selection
- **WHEN** `analyze_structure(filepath, detector="energy")` is called
- **THEN** uses the specified detector regardless of MSAF availability

#### Scenario: Auto detector selection
- **WHEN** `analyze_structure(filepath, detector="auto")` is called (default)
- **THEN** uses MSAF if available, otherwise falls back to energy-based detection

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

