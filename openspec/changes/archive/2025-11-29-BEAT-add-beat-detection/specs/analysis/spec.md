## ADDED Requirements

### Requirement: Beat Grid

The system SHALL provide a `BeatGrid` dataclass using index-based representation for compact storage and deterministic beat position regeneration.

#### Scenario: BeatGrid structure
- **WHEN** a `BeatGrid` is created
- **THEN** it contains `first_beat_time` (float), `bpm` (float), `time_signature` (tuple), `confidence` (float), and `method` (string)

#### Scenario: First downbeat accessor
- **WHEN** accessing `BeatGrid.first_downbeat`
- **THEN** returns `first_beat_time` (the anchor point where bar 1 begins)

#### Scenario: Generate beat timestamps
- **WHEN** calling `BeatGrid.to_beat_times(duration)`
- **THEN** returns array of beat timestamps computed from grid parameters

#### Scenario: Generate downbeat timestamps
- **WHEN** calling `BeatGrid.to_downbeat_times(duration)`
- **THEN** returns array of downbeat timestamps (every Nth beat based on time signature)

#### Scenario: Beat index to time conversion
- **WHEN** calling `BeatGrid.beat_to_time(beat_index)`
- **THEN** returns timestamp for that beat index

#### Scenario: Time to beat index conversion
- **WHEN** calling `BeatGrid.time_to_beat(time)`
- **THEN** returns beat index (float) for that timestamp

### Requirement: Beat Detection with beat_this

The system SHALL detect beat grid parameters using beat_this neural network.

#### Scenario: Detect beats with beat_this
- **WHEN** `detect_beats()` is called with a valid audio filepath
- **THEN** returns `BeatGrid` with parameters derived from beat_this inference

#### Scenario: BPM derived from beat intervals
- **WHEN** beat_this returns beat timestamps
- **THEN** BPM is calculated from median beat interval

#### Scenario: First beat time from downbeats
- **WHEN** beat_this returns downbeat timestamps
- **THEN** `first_beat_time` is set to first detected downbeat

#### Scenario: CUDA device selection
- **WHEN** CUDA is available on the system
- **THEN** beat_this uses GPU acceleration for faster inference

#### Scenario: CPU fallback
- **WHEN** CUDA is not available
- **THEN** beat_this falls back to CPU inference without errors

#### Scenario: beat_this import failure fallback
- **WHEN** beat_this is not installed or import fails
- **THEN** `detect_beats()` falls back to librosa-based detection

### Requirement: Beat Detection with librosa

The system SHALL provide librosa-based beat detection as a fallback method.

#### Scenario: Detect beats with librosa
- **WHEN** `detect_beats_librosa()` is called
- **THEN** returns `BeatGrid` with parameters derived from librosa

#### Scenario: First beat estimation from energy
- **WHEN** librosa beat detection completes
- **THEN** estimates first downbeat using beat energy analysis (highest energy beat in 4-beat groups)

#### Scenario: Pre-loaded audio support
- **WHEN** `detect_beats_librosa()` is called with `audio` parameter
- **THEN** uses provided audio data instead of loading from disk

### Requirement: Beat Grid from Timestamps

The system SHALL support creating a `BeatGrid` from raw timestamp arrays for interoperability.

#### Scenario: Create from beat timestamps
- **WHEN** calling `BeatGrid.from_timestamps(beats, downbeats)`
- **THEN** derives `first_beat_time`, `bpm`, and `time_signature` from the arrays

#### Scenario: BPM from median interval
- **WHEN** beat timestamps are provided
- **THEN** BPM is calculated from median beat interval for robustness

### Requirement: First Downbeat Annotations

Ground truth annotation files SHALL support specifying the first downbeat position for accurate bar alignment.

#### Scenario: First downbeat column in CSV
- **WHEN** loading annotations from CSV
- **THEN** accept optional `first_downbeat` column with time in seconds

#### Scenario: First downbeat used for bar calculations
- **WHEN** annotation includes `first_downbeat` value
- **THEN** bar positions in that file are calculated relative to the specified first downbeat

#### Scenario: Missing first downbeat defaults to zero
- **WHEN** annotation file lacks `first_downbeat` column
- **THEN** assume first_downbeat=0.0 for backward compatibility

#### Scenario: Per-track first downbeat
- **WHEN** multiple tracks in annotation file have different first downbeat values
- **THEN** each track uses its own first_downbeat for bar calculations
