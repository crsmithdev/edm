# Change: Implement BPM Analysis with Cascading Lookup Strategy

## Why
The current implementation has only placeholder/stub code for BPM detection. Users need actual working BPM analysis that can intelligently use multiple sources (file metadata, online APIs, computed analysis) to provide the most accurate and efficient results. This is a core feature required for DJ workflow integration.

## What Changes
- **Implement BPM metadata extraction** from audio files (ID3, MP4, FLAC tags)
- **Implement Spotify API integration** for fetching BPM from Spotify's audio features
- **Implement computed BPM detection** using madmom (primary) and librosa (fallback)
- **Add cascading lookup strategy** that tries: local metadata → Spotify API → computed analysis
- **Add configuration options** for controlling lookup strategy and analysis parameters
- **Update CLI** to show BPM source (metadata/spotify/computed) in output

## Impact
- Affected specs: `core-library`, `cli`
- Affected code:
  - `src/edm/analysis/bpm.py` - Replace placeholder with real implementation
  - `src/edm/io/metadata.py` - Add BPM extraction from file tags
  - `src/edm/external/spotify.py` - Implement Spotify API calls
  - `src/edm/config.py` - Add BPM lookup strategy configuration
  - `src/cli/commands/analyze.py` - Show BPM source in output
- New code:
  - `src/edm/analysis/bpm_detector.py` - Madmom/librosa implementation
  - Tests for all new functionality
