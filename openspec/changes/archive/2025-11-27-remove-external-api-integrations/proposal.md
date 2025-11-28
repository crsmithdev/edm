# Change: Remove External API Integrations

## Why

External BPM/metadata APIs have proven unreliable for programmatic access:

1. **Spotify** - Audio Features API deprecated December 2024
2. **GetSongBPM** - Blocked by Cloudflare bot protection, unusable for server-side requests
3. **Beatport** - No public API, requires web scraping
4. **TuneBat** - No public API

Rather than chasing unreliable external services, the tool should rely on:
- **File metadata** - Already embedded in most professionally tagged files
- **Local audio analysis** - `beat_this` neural network detection is accurate and works offline

This simplifies the architecture, removes external dependencies, and makes the tool fully self-contained.

## What Changes

1. **Remove `src/edm/external/` module entirely**
   - Delete `spotify.py`, `getsongbpm.py`, `__init__.py`
   - Remove all external API client code

2. **Simplify BPM strategy**
   - Change from `metadata → external → computed`
   - To: `metadata → computed`
   - Remove external lookup step entirely

3. **Update configuration**
   - Remove `ExternalServicesConfig` class
   - Remove `getsongbpm_api_key`, `spotify_client_id`, `spotify_client_secret`
   - Simplify `bpm_lookup_strategy` options

4. **Remove dependencies**
   - Remove `spotipy` from dependencies
   - Remove `python-dotenv` (only needed for API keys)

5. **Delete obsolete proposal**
   - Remove `replace-spotify-api` change (superseded by this)

6. **Update documentation**
   - Remove external API setup instructions
   - Document simplified metadata → computed flow

## Impact

- **Affected specs**: `core-library` (remove External Data Retrieval Module requirement)
- **Affected code**:
  - Deleted: `src/edm/external/` (entire directory)
  - Modified: `src/edm/analysis/bpm.py` (remove external lookups)
  - Modified: `src/edm/config.py` (remove external services config)
  - Modified: `pyproject.toml` (remove spotipy, python-dotenv)
- **Breaking change**: Users relying on Spotify/external lookups will need to use metadata or computed BPM
- **User benefit**: No API keys required, works fully offline, simpler setup
