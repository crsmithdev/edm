# Implementation Tasks

## 1. BPM Metadata Extraction
- [x] 1.1 Install and configure mutagen library for reading audio file tags
- [x] 1.2 Implement BPM extraction from ID3v2 tags (MP3)
- [x] 1.3 Implement BPM extraction from MP4 tags (M4A, AAC)
- [x] 1.4 Implement BPM extraction from FLAC vorbis comments
- [x] 1.5 Handle missing or invalid BPM metadata gracefully
- [x] 1.6 Add logging for metadata read operations

## 2. Spotify API Integration
- [x] 2.1 Implement OAuth2 authentication flow with spotipy
- [x] 2.2 Implement track search by artist and title
- [x] 2.3 Implement audio features API call to get BPM
- [x] 2.4 Add response caching to avoid redundant API calls
- [x] 2.5 Handle API rate limiting and errors gracefully
- [x] 2.6 Add configuration for Spotify credentials (client ID/secret)
- [x] 2.7 Support reading credentials from environment variables

## 3. Computed BPM Detection
- [x] 3.1 Implement madmom-based BPM detection (DBN tracker)
- [x] 3.2 Implement librosa-based BPM detection as fallback
- [x] 3.3 Handle tempo multiplicity (e.g., 128 vs 64 vs 256 BPM)
- [x] 3.4 Add confidence scoring for computed BPM
- [x] 3.5 Optimize performance for long audio files
- [x] 3.6 Add logging for detection steps and timing

## 4. Cascading Lookup Strategy
- [x] 4.1 Implement BPM lookup orchestration function
- [x] 4.2 Add strategy: try metadata first
- [x] 4.3 Add strategy: try Spotify API second
- [x] 4.4 Add strategy: compute as last resort
- [x] 4.5 Track and return BPM source in result
- [x] 4.6 Add configuration to customize lookup order
- [x] 4.7 Add option to skip certain sources
- [x] 4.8 Add option to force computation even if metadata/API exists (via flag composition)

## 5. CLI Updates
- [x] 5.1 Update analyze command output to show BPM source
- [x] 5.2 Add `--offline` flag to skip network calls (Spotify API)
- [x] 5.3 Add `--ignore-metadata` flag to skip file metadata lookups
- [x] 5.4 Implement flag composability (--offline + --ignore-metadata = compute only)
- [x] 5.5 Add visual indicators (icons/colors) for different BPM sources
- [x] 5.6 Update JSON output to include BPM source field
- [x] 5.7 Add timing breakdown showing time spent on each lookup attempt
- [x] 5.8 Show active strategy in verbose mode based on flags

## 6. Error Handling & Edge Cases
- [x] 7.1 Handle files with no metadata
- [x] 7.2 Handle invalid/corrupted metadata BPM values
- [x] 7.3 Handle Spotify API unavailable or credentials not configured
- [x] 7.4 Handle tracks not found on Spotify
- [x] 7.5 Handle ambiguous BPM results (e.g., 128 vs 64)
- [x] 7.6 Add validation for reasonable BPM range (40-200 for EDM)

## 8. Testing
- [x] 8.1 Create test fixtures with known BPM values (click tracks, beat patterns)
- [x] 8.2 Test BPM computation accuracy on synthetic tracks (test_analysis.py)
- [x] 8.3 Test metadata extraction for multiple formats (test_metadata.py)
- [x] 8.4 Test cascading strategy with success/failure paths (test_analysis.py)
- [x] 8.5 Integration test CLI commands (test_cli.py)
