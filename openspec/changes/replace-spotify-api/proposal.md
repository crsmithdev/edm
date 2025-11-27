# Change: Replace Spotify API

## Why

Spotify's Audio Features API has been deprecated/discontinued (reported February 2025). The API endpoints that provided BPM, key, energy, and danceability data are no longer reliably available, even for existing applications. This breaks the current BPM lookup strategy which relies on Spotify as a fallback when metadata is unavailable.

The cascading strategy (`metadata → spotify → computed`) currently depends on Spotify for intermediate BPM lookups, which account for a significant portion of successful lookups when file metadata is missing or incomplete. Without a replacement, users will experience:
- Increased analysis time (more tracks falling through to computation)
- Loss of curated BPM data from professional sources
- Inconsistent results compared to DJ software that uses external databases

## What Changes

Replace Spotify API integration with alternative music database sources:

1. **Implement Beatport client** - EDM-focused database with accurate BPM and key data
   - Web scraping implementation (no official API)
   - Search by artist + title
   - Extract BPM, key, genre from track pages
   - Rate limiting and respectful scraping

2. **Implement TuneBat client** - Music analysis database with BPM and key data
   - Web scraping or API implementation (investigate available options)
   - Search by artist + title
   - Extract BPM, key, camelot key from results
   - Caching to minimize requests

3. **Deprecate Spotify client** - Mark as deprecated, provide migration path
   - Add deprecation warnings to SpotifyClient
   - Update BPM strategy to prefer new sources
   - Keep implementation for backward compatibility (existing cached data)
   - Document removal timeline

4. **Update BPM analysis strategy**
   - Change from `metadata → spotify → computed`
   - To: `metadata → beatport → tunebat → computed`
   - Configuration option to customize source order
   - Parallel lookup option (query multiple sources simultaneously)

5. **Update configuration**
   - Remove Spotify-specific config options (client ID, client secret)
   - Add Beatport/TuneBat settings (rate limits, cache TTL)
   - Add source preference configuration

## Impact

- **Affected specs**: `core-library` (External Data Retrieval Module requirement modifications)
- **Affected code**:
  - Modified: `src/edm/external/beatport.py` (implement web scraper)
  - Modified: `src/edm/external/tunebat.py` (implement web scraper/API)
  - Modified: `src/edm/external/spotify.py` (add deprecation warnings)
  - Modified: `src/edm/analysis/bpm.py` (update strategy, replace spotify calls)
  - Modified: `src/edm/config.py` (remove Spotify config, add new source config)
  - Modified: `docs/` (update API documentation, migration guide)
- **Dependencies**:
  - Add: `beautifulsoup4` and `requests` for web scraping
  - Remove: `spotipy` (optional, can keep for backward compat)
- **User experience**: Users will need to update configurations (remove Spotify credentials)
- **Backward compatibility**: Breaking change - Spotify-based lookups will be deprecated

## Design Decisions

### Why Beatport + TuneBat?

1. **Beatport** - Industry-standard EDM database
   - High accuracy for electronic music
   - Curated by professional DJs and labels
   - Includes genre, key, BPM for most EDM tracks
   - Well-suited for the target user base

2. **TuneBat** - Complementary coverage
   - Broader genre coverage beyond EDM
   - Free access via web interface
   - Community-contributed data with good accuracy
   - Fills gaps where Beatport doesn't have data

### Why not other alternatives?

- **Last.fm** - No BPM data in API
- **MusicBrainz** - Limited BPM coverage, inconsistent data
- **AcoustID/Chromaprint** - Audio fingerprinting only, no BPM database
- **ReccoBeats API** - Third-party service with unknown reliability/longevity

### Web Scraping vs API

Both Beatport and TuneBat lack official public APIs, requiring web scraping:
- Implement respectful scraping (rate limiting, user agent, caching)
- Follow robots.txt guidelines
- Add fallback to computation if scraping fails
- Document scraping limitations and legal considerations

### Migration Path

1. **Phase 1 (Immediate)**: Implement new sources, deprecate Spotify
2. **Phase 2 (v0.3.0)**: Remove Spotify dependency, update defaults
3. **Phase 3 (v0.4.0)**: Remove Spotify code entirely

Users can continue using Spotify temporarily by keeping credentials configured, but will receive deprecation warnings.
