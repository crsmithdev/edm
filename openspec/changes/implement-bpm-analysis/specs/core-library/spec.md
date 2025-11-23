# Core Library Specification Changes

## MODIFIED Requirements

### Requirement: Audio Analysis Module
The library SHALL provide an analysis module (`edm.analysis`) that contains functions for analyzing EDM tracks, including BPM detection with cascading lookup strategy.

#### Scenario: Analyze track for BPM using cascading strategy
- **WHEN** user calls `analyze_bpm(filepath)` with default configuration
- **THEN** system attempts to read BPM from file metadata first, then Spotify API, then computes it, returning the first successful result

#### Scenario: Force BPM computation
- **WHEN** user calls `analyze_bpm(filepath, force_compute=True)`
- **THEN** system skips metadata and API lookups and computes BPM directly

#### Scenario: BPM result includes source
- **WHEN** user calls `analyze_bpm(filepath)`
- **THEN** result includes the source of the BPM ("metadata", "spotify", or "computed")

#### Scenario: Handle invalid metadata BPM
- **WHEN** file metadata contains invalid BPM (e.g., 0, negative, > 300)
- **THEN** system logs warning and falls back to next lookup strategy

## MODIFIED Requirements

### Requirement: File I/O Module
The library SHALL provide an I/O module (`edm.io`) for reading and writing audio files and metadata, including BPM extraction from file tags.

#### Scenario: Read BPM from MP3 metadata
- **WHEN** user calls `read_metadata(filepath)` on MP3 file with BPM tag
- **THEN** returns metadata dictionary including BPM value from ID3v2 tags

#### Scenario: Read BPM from FLAC metadata
- **WHEN** user calls `read_metadata(filepath)` on FLAC file with BPM tag
- **THEN** returns metadata dictionary including BPM value from vorbis comments

#### Scenario: Read BPM from M4A metadata
- **WHEN** user calls `read_metadata(filepath)` on M4A file with BPM tag
- **THEN** returns metadata dictionary including BPM value from MP4 tags

#### Scenario: Handle missing BPM in metadata
- **WHEN** file metadata does not contain BPM field
- **THEN** returns metadata dictionary with BPM as None without raising error

## MODIFIED Requirements

### Requirement: External Data Retrieval Module
The library SHALL provide an external module (`edm.external`) for retrieving BPM and track information from Spotify with OAuth2 authentication and response caching.

#### Scenario: Authenticate with Spotify using credentials
- **WHEN** user initializes `SpotifyClient(client_id, client_secret)`
- **THEN** client obtains OAuth2 access token from Spotify API

#### Scenario: Search track and fetch BPM
- **WHEN** user calls `client.search_track(artist, title)` and track is found
- **THEN** returns `SpotifyTrackInfo` with BPM from audio features API

#### Scenario: Match track using artist and title
- **WHEN** user searches for track with exact artist and title match
- **THEN** returns the best matching track with high confidence

#### Scenario: Handle track not found on Spotify
- **WHEN** user searches for track that doesn't exist on Spotify
- **THEN** returns None without raising exception

#### Scenario: Cache Spotify API responses
- **WHEN** same track is queried multiple times
- **THEN** subsequent requests use cached response without API call

#### Scenario: Handle Spotify API rate limiting
- **WHEN** Spotify API returns 429 rate limit error
- **THEN** implements exponential backoff and retries request

#### Scenario: Read credentials from environment variables
- **WHEN** SpotifyClient is initialized without parameters
- **THEN** reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from environment

## ADDED Requirements

### Requirement: BPM Computation with madmom
The library SHALL provide BPM computation using madmom as the primary detector with librosa as fallback.

#### Scenario: Compute BPM using madmom
- **WHEN** user calls `compute_bpm(filepath, method="madmom")`
- **THEN** uses madmom DBN beat tracker to detect BPM with confidence score

#### Scenario: Compute BPM using librosa fallback
- **WHEN** madmom computation fails or user specifies `method="librosa"`
- **THEN** uses librosa tempo detection to compute BPM

#### Scenario: Handle tempo multiplicity
- **WHEN** track has ambiguous tempo (e.g., could be 128 or 64 BPM)
- **THEN** returns primary BPM in EDM range (120-150) with alternative tempos in metadata

#### Scenario: Validate BPM range
- **WHEN** computed BPM is outside valid range (40-200 for EDM)
- **THEN** logs warning and returns result with low confidence score

## MODIFIED Requirements

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports BPM lookup strategy configuration with file-based and programmatic options.

#### Scenario: Configure BPM lookup order
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "spotify", "computed"]`
- **THEN** BPM analysis follows specified order

#### Scenario: Skip specific lookup sources
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "computed"]`
- **THEN** system skips Spotify API and only uses metadata and computation

#### Scenario: Force computation via configuration
- **WHEN** user sets `config.bpm_force_compute = True`
- **THEN** all BPM lookups skip metadata and API, computing directly

#### Scenario: Configure Spotify cache TTL
- **WHEN** user sets `config.external_services.cache_ttl = 7200`
- **THEN** Spotify API responses are cached for 2 hours

## ADDED Requirements

### Requirement: BPM Result Transparency
The library SHALL include source attribution in all BPM results for transparency and debugging.

#### Scenario: Result shows metadata source
- **WHEN** BPM is obtained from file metadata
- **THEN** result includes `source="metadata"` and `method` is None

#### Scenario: Result shows Spotify source
- **WHEN** BPM is obtained from Spotify API
- **THEN** result includes `source="spotify"` and `method` is None

#### Scenario: Result shows computed source
- **WHEN** BPM is computed using madmom
- **THEN** result includes `source="computed"` and `method="madmom-dbn"`

#### Scenario: Result includes computation time
- **WHEN** BPM is computed (not from metadata or API)
- **THEN** result includes `computation_time` in seconds

## ADDED Requirements

### Requirement: Batch Processing Optimization
The library SHALL optimize BPM analysis for batch processing with caching and parallel support.

#### Scenario: Cache prevents redundant API calls
- **WHEN** analyzing multiple files from same artist/album
- **THEN** Spotify API is called once per unique track, using cache for duplicates

#### Scenario: Parallel metadata reading
- **WHEN** analyzing batch of 100+ files
- **THEN** metadata reading happens in parallel for improved performance

#### Scenario: Progress callback for batch operations
- **WHEN** analyzing batch with progress callback provided
- **THEN** callback is invoked with progress updates after each track
