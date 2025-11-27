# core-library Specification Delta

## Modified Requirements

### Requirement: External Data Retrieval Module
The library SHALL provide an external module (`edm.external`) for retrieving BPM and track information from Beatport, TuneBat, and optionally Spotify (deprecated).

#### Scenario: Query Beatport for BPM
- **WHEN** user calls `search_beatport(artist, title)` with valid parameters
- **THEN** returns BPM, key, and genre information from Beatport

#### Scenario: Query TuneBat for BPM
- **WHEN** user calls `search_tunebat(artist, title)` with valid parameters
- **THEN** returns BPM, key, and camelot key data from TuneBat

#### Scenario: Query Spotify for track info (DEPRECATED)
- **WHEN** user calls `search_spotify(artist, title)` with valid parameters
- **THEN** returns matching track information including BPM if available from Spotify API
- **AND** logs deprecation warning indicating Spotify will be removed in future version

#### Scenario: Handle web scraping errors
- **WHEN** web scraping request fails due to HTTP error or parsing error
- **THEN** raises a custom `ExternalServiceError` with error details and retry information

#### Scenario: Rate limiting for web scrapers
- **WHEN** making multiple requests to same external source
- **THEN** enforces rate limiting to respect source's robots.txt and prevent service disruption

#### Scenario: Cache external requests
- **WHEN** the same external data request is made within cache lifetime
- **THEN** returns cached result without making new API call or web scrape

#### Scenario: Aggregate results from multiple sources
- **WHEN** user calls `get_track_info(artist, title)` without specifying source
- **THEN** queries available sources (Beatport, TuneBat) in configured order and returns first successful result with source attribution

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports BPM lookup strategy configuration with file-based and programmatic options.

#### Scenario: Configure BPM lookup order
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "beatport", "tunebat", "computed"]`
- **THEN** BPM analysis follows specified order

#### Scenario: Skip specific lookup sources
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "computed"]`
- **THEN** system skips external APIs and only uses metadata and computation

#### Scenario: Force computation via configuration
- **WHEN** user sets `config.bpm_force_compute = True`
- **THEN** all BPM lookups skip metadata and API, computing directly

#### Scenario: Configure external service cache TTL
- **WHEN** user sets `config.external_services.cache_ttl = 7200`
- **THEN** external service responses (Beatport, TuneBat) are cached for 2 hours

#### Scenario: Configure rate limiting
- **WHEN** user sets `config.external_services.beatport.rate_limit = 1.0`
- **THEN** Beatport requests are rate limited to 1 request per second

#### Scenario: Use deprecated Spotify configuration
- **WHEN** user has Spotify credentials configured in legacy configuration
- **THEN** Spotify client continues to work but logs deprecation warnings
