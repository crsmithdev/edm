# Design: BPM Analysis with Cascading Lookup Strategy

## Context
The current implementation has stub/placeholder code for BPM detection. We need to implement a production-ready BPM analysis system that prioritizes efficiency and accuracy by using multiple data sources in a cascading manner.

**User Goal:** Get accurate BPM values quickly, preferring existing metadata over API calls, and only computing when necessary.

**System Constraints:**
- Spotify API has rate limits (no more than a few requests per second)
- Computing BPM is expensive (can take 5-30 seconds per track)
- File metadata is instant but may be missing or incorrect
- Users may analyze hundreds of tracks in batch

## Goals / Non-Goals

**Goals:**
- Implement cascading BPM lookup: metadata → Spotify → computation
- Make strategy configurable for different use cases
- Track and report BPM source for transparency
- Handle errors gracefully at each stage
- Optimize for batch processing (caching, parallelization)

**Non-Goals:**
- Other online services besides Spotify (defer to future changes)
- Beat grid alignment (covered by separate feature)
- Key detection or harmonic analysis
- Real-time BPM tracking for live input

## Decisions

### Decision 1: Cascading Lookup Strategy
**What:** Try BPM sources in order: file metadata → Spotify API → computed analysis. Stop at first successful result.

**Why:**
- **Metadata is instant** (< 1ms) and often accurate for properly tagged files
- **Spotify API is fast** (100-500ms) and has professional-grade analysis
- **Computation is slow** (5-30s) but most accurate for untagged files
- Cascading maximizes speed while ensuring we always get a result

**Alternatives considered:**
- Always compute: Too slow for batch processing
- Only use metadata: Fails for untagged files
- Parallel lookup with voting: Wastes resources and increases complexity

**Implementation:**
```python
async def get_bpm(filepath: Path, config: BPMConfig) -> BPMResult:
    """Get BPM using cascading lookup strategy."""
    for source in config.lookup_strategy:
        try:
            if source == "metadata":
                bpm = await read_bpm_from_metadata(filepath)
                if is_valid_bpm(bpm):
                    return BPMResult(bpm=bpm, source="metadata", confidence=0.7)
            
            elif source == "spotify":
                bpm = await fetch_bpm_from_spotify(filepath)
                if bpm:
                    return BPMResult(bpm=bpm, source="spotify", confidence=0.9)
            
            elif source == "computed":
                bpm = await compute_bpm(filepath)
                return BPMResult(bpm=bpm, source="computed", confidence=0.95)
        
        except Exception as e:
            logger.warning(f"BPM lookup failed for {source}: {e}")
            continue
    
    raise AnalysisError("All BPM lookup strategies failed")
```

### Decision 2: Use madmom as Primary Detector
**What:** Use madmom's DBNBeatTracker as the primary BPM computation method, with librosa as fallback.

**Why:**
- Madmom is specifically trained on electronic music
- DBN (Dynamic Bayesian Network) tracker handles EDM better than autocorrelation
- Proven accuracy in academic benchmarks
- Handles tempo multiplicity (128 vs 64 vs 256 BPM)

**Alternatives considered:**
- Librosa only: Less accurate for EDM
- Essentia: Good but heavier dependency
- Custom ML model: Premature at this stage

**Configuration:**
```python
class BPMComputeConfig(BaseModel):
    prefer_madmom: bool = True  # Try madmom first
    madmom_fps: int = 100  # Frames per second for madmom
    librosa_hop_length: int = 512  # Hop length for librosa
    min_bpm: float = 40.0  # Minimum valid BPM
    max_bpm: float = 200.0  # Maximum valid BPM
```

### Decision 3: Track BPM Source in Results
**What:** Always include the source of BPM data in the result object.

**Why:**
- **Transparency:** Users know where the data came from
- **Debugging:** Easier to troubleshoot incorrect BPMs
- **Trust:** Users can decide if they trust computed vs API vs metadata
- **Analytics:** Track which sources are most commonly used

**Result Schema:**
```python
@dataclass
class BPMResult:
    bpm: float
    confidence: float
    source: Literal["metadata", "spotify", "computed"]
    method: Optional[str] = None  # e.g., "madmom-dbn", "librosa-autocorr"
    computation_time: float = 0.0  # Seconds spent computing
```

### Decision 4: Spotify Track Matching Strategy
**What:** Match tracks to Spotify using artist + title from file metadata, with fuzzy matching fallback.

**Why:**
- Most accurate matching uses both artist and title
- File metadata provides these fields for most files
- Fuzzy matching handles slight variations (featuring artists, remixes)

**Matching Algorithm:**
1. Try exact match on `{artist} - {title}`
2. Try fuzzy match with threshold > 0.85
3. Try search with first result if high confidence
4. Return None if no good match found

**Implementation:**
```python
async def match_spotify_track(metadata: AudioMetadata) -> Optional[str]:
    """Match track to Spotify and return track ID."""
    # Exact search
    query = f"{metadata.artist} {metadata.title}"
    results = spotify.search(q=query, type="track", limit=5)
    
    # Find best match using string similarity
    best_match = None
    best_score = 0.0
    
    for track in results['tracks']['items']:
        score = calculate_similarity(track, metadata)
        if score > best_score:
            best_score = score
            best_match = track
    
    if best_score > 0.85:
        return best_match['id']
    
    return None
```

### Decision 5: Caching Strategy
**What:** Cache Spotify API responses in-memory with LRU eviction and file-based persistence.

**Why:**
- **Reduces API calls:** Same track analyzed multiple times uses cached data
- **Batch optimization:** Re-analyzing same directory doesn't hit API again
- **Rate limit protection:** Avoid hitting Spotify's rate limits
- **Offline support:** Can work with previously cached data

**Cache Structure:**
```python
class SpotifyCache:
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400):
        self.memory_cache = LRU(maxsize=1000)
        self.cache_file = cache_dir / "spotify_cache.json"
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[dict]:
        # Try memory first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try disk cache
        cached = self._load_from_disk(key)
        if cached and not self._is_expired(cached):
            self.memory_cache[key] = cached['data']
            return cached['data']
        
        return None
```

## Risks / Trade-offs

**Risk: Incorrect metadata BPM**
- Files may have wrong BPM in tags
- **Mitigation:** Add `--force-compute` flag, validate BPM range (40-200), allow configuration to skip metadata

**Risk: Spotify API rate limits**
- Batch processing could hit rate limits
- **Mitigation:** Implement exponential backoff, caching, option to disable Spotify

**Risk: Tempo multiplicity ambiguity**
- 128 BPM track might be detected as 64 or 256
- **Mitigation:** Use madmom's multiple tempo detection, prefer values in EDM range (120-150)

**Risk: Computation too slow for large batches**
- Computing 1000 tracks could take hours
- **Mitigation:** Parallel processing, progress bars, option to skip computation

### Decision 6: CLI Flag Design for Cascading Control
**What:** Provide simple, composable flags to control the cascading lookup strategy.

**Why:**
- Users need easy control over which sources to use
- Common cases should be simple (one flag)
- Advanced users can combine flags for fine control
- Follows Unix/GNU conventions

**Flag Design:**

**Default behavior (no flags):**
```bash
edm analyze track.mp3
# Strategy: metadata → spotify → computed
```

**Primary flags:**
- `--offline` - Skip all network calls (Spotify API)
  - Strategy becomes: metadata → computed
- `--ignore-metadata` - Skip file metadata lookups
  - Strategy becomes: spotify → computed

**Composability:**
```bash
# Force computation by combining flags
edm analyze track.mp3 --offline --ignore-metadata
# Strategy becomes: computed only

# Only use Spotify (fail if not found)
edm analyze track.mp3 --ignore-metadata --no-compute
```

**Configuration file equivalents:**
```toml
[bpm]
# Default strategy
lookup_strategy = ["metadata", "spotify", "computed"]

# Skip metadata
lookup_strategy = ["spotify", "computed"]

# Offline mode
lookup_strategy = ["metadata", "computed"]

# Force computation
lookup_strategy = ["computed"]
```

**Priority order:**
1. CLI flags (highest priority)
2. Environment variables
3. Configuration file
4. Defaults

**Alternatives considered:**
- `--no-metadata`, `--no-spotify`, `--no-compute`: More verbose, harder to remember
- `--strategy metadata,spotify,compute`: Too complex for common cases
- `--from metadata`: Unclear if it's exclusive or preferred

## Migration Plan

**Phase 1: Core Implementation**
1. Implement metadata reading with mutagen
2. Implement Spotify API integration with spotipy
3. Implement madmom BPM computation
4. Add cascading strategy orchestration

**Phase 2: Configuration & CLI**
1. Add configuration options for lookup strategy
2. Update CLI to show BPM source
3. Add `--offline` and `--ignore-metadata` flags
4. Update output formatting

**Phase 3: Optimization**
1. Add caching layer for Spotify
2. Implement parallel processing for batches
3. Performance tuning for madmom

**Rollback Plan:**
- Keep placeholder code in separate module
- Feature flag to switch between stub and real implementation
- If issues arise, disable Spotify API or computation via config

## Open Questions

1. **Should we support other services (Beatport, TuneBat) in this change?**
   - **Decision:** No, focus on Spotify first. Other services in future change.

2. **How to handle ambiguous BPM (e.g., 128 vs 256)?**
   - **Decision:** Prefer values in EDM range (120-150), expose all candidates in debug mode.

3. **Should we validate computed BPM against Spotify/metadata?**
   - **Decision:** Not in this change. Validation/correction is a future feature.

4. **Parallel processing for batch analysis?**
   - **Decision:** Add basic async support, full parallelization in future optimization change.
