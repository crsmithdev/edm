## ADDED Requirements

### Requirement: BPM Source Enumeration

The system SHALL use a typed enumeration for BPM source identification instead of string literals.

#### Scenario: Enum values match legacy strings
- **WHEN** `BPMSource.METADATA` is converted to string
- **THEN** the result is `"metadata"` for backward compatibility

#### Scenario: Type-safe source comparison
- **WHEN** comparing BPM result sources in application code
- **THEN** IDE autocomplete and type checking prevent typos like `"metdata"`

### Requirement: Instance-scoped API Caching

The SpotifyClient SHALL maintain caches that are scoped to individual instances rather than shared across all instances.

#### Scenario: Separate cache per client instance
- **WHEN** two SpotifyClient instances are created
- **THEN** each instance maintains its own independent cache

#### Scenario: Cache cleared on instance
- **WHEN** `client.clear_cache()` is called on one instance
- **THEN** only that instance's cache is cleared, other instances unaffected
