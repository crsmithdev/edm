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

