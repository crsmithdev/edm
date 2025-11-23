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

## 5. Configuration Updates
- [ ] 5.1 Add `bpm_lookup_strategy` config option (default: ["metadata", "spotify", "computed"])
- [ ] 5.2 Add `bpm_force_compute` config option (default: False)
- [ ] 5.3 Add `bpm_prefer_madmom` config option (default: True)
- [ ] 5.4 Add Spotify credentials to external services config
- [ ] 5.5 Add cache configuration for API responses

## 6. CLI Updates
- [x] 6.1 Update analyze command output to show BPM source
- [x] 6.2 Add `--offline` flag to skip network calls (Spotify API)
- [x] 6.3 Add `--ignore-metadata` flag to skip file metadata lookups
- [x] 6.4 Implement flag composability (--offline + --ignore-metadata = compute only)
- [x] 6.5 Add visual indicators (icons/colors) for different BPM sources
- [x] 6.6 Update JSON output to include BPM source field
- [x] 6.7 Add timing breakdown showing time spent on each lookup attempt
- [x] 6.8 Show active strategy in verbose mode based on flags

## 7. Error Handling & Edge Cases
- [x] 7.1 Handle files with no metadata
- [x] 7.2 Handle invalid/corrupted metadata BPM values
- [x] 7.3 Handle Spotify API unavailable or credentials not configured
- [x] 7.4 Handle tracks not found on Spotify
- [x] 7.5 Handle ambiguous BPM results (e.g., 128 vs 64)
- [x] 7.6 Add validation for reasonable BPM range (40-200 for EDM)

## 8. Testing
- [ ] 8.1 Create test fixtures with known BPM values in metadata
- [ ] 8.2 Mock Spotify API responses for unit tests
- [ ] 8.3 Test cascading strategy with all success paths
- [ ] 8.4 Test cascading strategy with all failure paths
- [ ] 8.5 Test BPM computation accuracy on sample tracks
- [ ] 8.6 Integration test full analyze command with real files
- [ ] 8.7 Test configuration override behavior
- [ ] 8.8 Performance test for batch analysis

## 9. Documentation
- [ ] 9.1 Document BPM lookup strategy in README
- [ ] 9.2 Document Spotify API setup instructions
- [ ] 9.3 Document configuration options for BPM analysis
- [ ] 9.4 Add examples of different lookup strategies
- [ ] 9.5 Document BPM source field in output
- [ ] 9.6 Update API documentation with new functions
