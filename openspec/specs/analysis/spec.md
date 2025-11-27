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

