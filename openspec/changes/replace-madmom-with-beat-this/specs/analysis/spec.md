## MODIFIED Requirements

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

## ADDED Requirements

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

## REMOVED Requirements

### Requirement: madmom BPM Detection

**Reason**: madmom is unmaintained (last release 2019), requires Python <3.10, and causes persistent installation failures due to Cython/NumPy version conflicts.

**Migration**: Replace with beat_this which provides equivalent or better accuracy from the same research group (CPJKU) with modern Python support.
