## REMOVED Requirements

### Requirement: External Data Retrieval Module
**Reason**: External BPM APIs are unreliable - Spotify deprecated, GetSongBPM blocked by Cloudflare, Beatport/TuneBat have no public APIs. Local audio analysis with beat_this provides accurate results without external dependencies.

**Migration**: Use file metadata or computed BPM. The `beat_this` neural network detector provides accurate BPM for EDM tracks.

## MODIFIED Requirements

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports BPM lookup strategy configuration with file-based and programmatic options.

#### Scenario: Configure BPM lookup order
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "computed"]`
- **THEN** BPM analysis follows specified order

#### Scenario: Skip metadata lookup
- **WHEN** user sets `config.bpm_lookup_strategy = ["computed"]`
- **THEN** system skips metadata and computes BPM directly from audio

#### Scenario: Force computation via configuration
- **WHEN** user sets `config.bpm_force_compute = True`
- **THEN** all BPM lookups skip metadata, computing directly from audio

### Requirement: Error Handling
The library SHALL define custom exception classes for different error categories.

#### Scenario: Audio file errors
- **WHEN** audio file cannot be loaded
- **THEN** raises `AudioFileError` with specific reason (not found, unsupported format, corrupted)

#### Scenario: Analysis errors
- **WHEN** analysis fails due to invalid input
- **THEN** raises `AnalysisError` with details about what failed
