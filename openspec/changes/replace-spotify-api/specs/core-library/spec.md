# core-library Specification Delta

## Modified Requirements

### Requirement: External Data Retrieval Module
The library SHALL provide an external module (`edm.external`) for retrieving BPM and track information from GetSongBPM API and optionally Spotify (deprecated).

#### Scenario: Query GetSongBPM for BPM
- **WHEN** user calls `search_getsongbpm(artist, title)` with valid parameters
- **THEN** returns BPM (tempo), key, and time signature from GetSongBPM API

#### Scenario: Query GetSongBPM by song ID
- **WHEN** user calls `get_song_by_id(song_id)` with valid GetSongBPM ID
- **THEN** returns full track metadata including BPM, key, danceability

#### Scenario: Query Spotify for track info (DEPRECATED)
- **WHEN** user calls `search_spotify(artist, title)` with valid parameters
- **THEN** returns matching track information including BPM if available from Spotify API
- **AND** logs deprecation warning indicating Spotify will be removed in future version

#### Scenario: Handle API errors
- **WHEN** GetSongBPM API request fails due to HTTP error, rate limit, or invalid API key
- **THEN** raises a custom `ExternalServiceError` with error details and retry information

#### Scenario: Rate limiting for API requests
- **WHEN** making multiple requests to GetSongBPM API
- **THEN** enforces configurable rate limiting to prevent exceeding API quotas

#### Scenario: Cache external requests
- **WHEN** the same external data request is made within cache lifetime
- **THEN** returns cached result without making new API call

#### Scenario: Handle missing API key
- **WHEN** GetSongBPM API key is not configured
- **THEN** skips GetSongBPM lookup and falls through to next source in strategy
- **AND** logs warning about missing API key

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports BPM lookup strategy configuration with file-based and programmatic options.

#### Scenario: Configure BPM lookup order
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "getsongbpm", "computed"]`
- **THEN** BPM analysis follows specified order

#### Scenario: Skip external lookup sources
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "computed"]`
- **THEN** system skips external APIs and only uses metadata and computation

#### Scenario: Force computation via configuration
- **WHEN** user sets `config.bpm_force_compute = True`
- **THEN** all BPM lookups skip metadata and API, computing directly

#### Scenario: Configure GetSongBPM API key
- **WHEN** user sets `config.external_services.getsongbpm_api_key = "abc123"`
- **THEN** GetSongBPM client uses provided API key for authentication

#### Scenario: Configure external service cache TTL
- **WHEN** user sets `config.external_services.cache_ttl = 7200`
- **THEN** external service responses are cached for 2 hours

#### Scenario: Configure rate limiting
- **WHEN** user sets `config.external_services.getsongbpm.rate_limit = 1.0`
- **THEN** GetSongBPM requests are rate limited to 1 request per second

#### Scenario: Use deprecated Spotify configuration
- **WHEN** user has Spotify credentials configured in legacy configuration
- **THEN** Spotify client continues to work but logs deprecation warnings

## Removed Requirements

### Requirement: Beatport Integration
The Beatport client stub SHALL be removed as Beatport has no public API.

#### Scenario: Beatport code removed
- **WHEN** user attempts to import BeatportClient
- **THEN** import fails as module no longer exists

### Requirement: TuneBat Integration
The TuneBat client stub SHALL be removed as TuneBat has no direct public API.

#### Scenario: TuneBat code removed
- **WHEN** user attempts to import TuneBatClient
- **THEN** import fails as module no longer exists

## Added Requirements

### Requirement: Attribution Display
The library SHALL display GetSongBPM attribution when BPM data is retrieved from their API, as required by their Terms of Service.

#### Scenario: Attribution in JSON output
- **WHEN** BPM result source is "getsongbpm"
- **THEN** JSON output includes `attribution` field with GetSongBPM link

#### Scenario: Attribution in CLI table
- **WHEN** CLI displays results with GetSongBPM-sourced BPM values
- **THEN** table footer includes "BPM data from getsongbpm.com" attribution
