# Change: Replace Spotify API with GetSongBPM

## Why

Spotify's Audio Features API has been deprecated/discontinued (December 2024). The API endpoints that provided BPM, key, energy, and danceability data are no longer available for new applications. This breaks the current BPM lookup strategy which relies on Spotify as a fallback when metadata is unavailable.

The cascading strategy (`metadata → spotify → computed`) currently depends on Spotify for intermediate BPM lookups, which account for a significant portion of successful lookups when file metadata is missing or incomplete. Without a replacement, users will experience:
- Increased analysis time (more tracks falling through to computation)
- Loss of curated BPM data from professional sources
- Inconsistent results compared to DJ software that uses external databases

## What Changes

Replace Spotify API integration with GetSongBPM API:

1. **Implement GetSongBPM client** - Free music database API with BPM and key data
   - REST API with JSON responses (no web scraping needed)
   - Search by artist + title or song ID
   - Returns BPM (tempo), key, time signature, danceability
   - Free tier with attribution requirement
   - Simple API key authentication

2. **Remove Beatport/TuneBat stubs** - Neither has a public API
   - Beatport requires web scraping (fragile, ToS concerns)
   - TuneBat has no direct API (partner API via Soundcharts is paid)
   - Remove placeholder implementations

3. **Deprecate Spotify client** - Mark as deprecated, provide migration path
   - Add deprecation warnings to SpotifyClient
   - Update BPM strategy to use GetSongBPM
   - Keep implementation for backward compatibility (existing users)
   - Document removal timeline

4. **Update BPM analysis strategy**
   - Change from `metadata → spotify → computed`
   - To: `metadata → getsongbpm → computed`
   - Simpler two-source external lookup
   - Configuration option to customize source order

5. **Update configuration**
   - Remove Spotify-specific config options (client ID, client secret)
   - Add GetSongBPM settings (API key, rate limits, cache TTL)
   - Add attribution link to CLI output (required by GetSongBPM ToS)

## Impact

- **Affected specs**: `core-library` (External Data Retrieval Module requirement modifications)
- **Affected code**:
  - New: `src/edm/external/getsongbpm.py` (implement API client)
  - Modified: `src/edm/external/spotify.py` (add deprecation warnings)
  - Removed: `src/edm/external/beatport.py` (remove stub)
  - Removed: `src/edm/external/tunebat.py` (remove stub)
  - Modified: `src/edm/analysis/bpm.py` (update strategy, replace spotify calls)
  - Modified: `src/edm/config.py` (remove Spotify config, add GetSongBPM config)
  - Modified: `docs/` (update API documentation, migration guide)
- **Dependencies**:
  - Add: `httpx` for async HTTP requests (already considering for other uses)
  - Remove: `spotipy` (can remove entirely or keep for backward compat)
- **User experience**: Users will need to:
  - Register for free GetSongBPM API key
  - Update configurations (remove Spotify credentials, add API key)
  - Add attribution link if distributing/publishing results
- **Backward compatibility**: Breaking change - Spotify-based lookups will be deprecated

## Design Decisions

### Why GetSongBPM?

1. **Free API with good coverage** - Large database including EDM tracks
2. **Actual REST API** - No web scraping required, reliable and stable
3. **Simple authentication** - Just an API key, no OAuth flow
4. **Returns all needed data** - BPM, key, time signature in one call
5. **Active and maintained** - Regular updates, responsive to developers

### Why not other alternatives?

- **Beatport** - No public API, requires fragile web scraping
- **TuneBat** - No direct API, partner API (Soundcharts) is paid
- **Last.fm** - No BPM data in API
- **MusicBrainz/AcousticBrainz** - Limited BPM coverage, inconsistent data
- **Cyanite** - Paid service, overkill for BPM-only needs
- **ReccoBeats** - Third-party wrapper with unknown reliability

### Attribution Requirement

GetSongBPM's free tier requires a backlink to getsongbpm.com. This will be handled by:
- Adding attribution to JSON output metadata
- Adding attribution note to CLI table footer
- Documenting requirement for users building on the library

### Migration Path

1. **Phase 1 (Immediate)**: Implement GetSongBPM, deprecate Spotify
2. **Phase 2 (v0.3.0)**: Remove Spotify dependency, update defaults
3. **Phase 3 (v0.4.0)**: Remove Spotify code and Beatport/TuneBat stubs entirely

Users can continue using Spotify temporarily by keeping credentials configured, but will receive deprecation warnings.
