# Implementation Tasks

## Phase 1: Implement GetSongBPM Client

### 1. GetSongBPM Client Implementation
- [ ] Create `src/edm/external/getsongbpm.py`
  - [ ] Define `GetSongBPMTrackInfo` dataclass (bpm, key, time_sig, title, artist)
  - [ ] Implement `GetSongBPMClient` class with API key authentication
  - [ ] Implement `search_track(artist, title)` method using search endpoint
  - [ ] Implement `get_song_by_id(song_id)` method for direct lookup
  - [ ] Add rate limiting (configurable, default 1 req/sec)
  - [ ] Add response caching with configurable TTL
  - [ ] Handle HTTP errors (401 unauthorized, 429 rate limit, 5xx server errors)
  - [ ] Return None gracefully when track not found

### 2. Add Dependencies
- [ ] Add `httpx` to pyproject.toml for async HTTP requests
- [ ] Update `uv.lock` with new dependencies
- [ ] Verify no conflicts with existing dependencies

### 3. Update Configuration
- [ ] Modify `src/edm/config.py`
  - [ ] Add `getsongbpm_api_key` field to ExternalServicesConfig
  - [ ] Add `getsongbpm_rate_limit` field (default: 1.0 req/sec)
  - [ ] Add deprecation notice to Spotify config fields
  - [ ] Update default BPM lookup strategy to `["metadata", "getsongbpm", "computed"]`
  - [ ] Add environment variable support: `GETSONGBPM_API_KEY`

### 4. Update BPM Analysis Strategy
- [ ] Modify `src/edm/analysis/bpm.py`
  - [ ] Add `_try_getsongbpm()` helper function
  - [ ] Update `analyze_bpm()` to use new strategy: metadata → getsongbpm → computed
  - [ ] Add deprecation warning when Spotify is used
  - [ ] Keep `_try_spotify()` for backward compatibility
  - [ ] Handle missing API key gracefully (skip to next source)

### 5. Update External Module
- [ ] Modify `src/edm/external/__init__.py`
  - [ ] Export `GetSongBPMClient` and `GetSongBPMTrackInfo`
  - [ ] Keep Spotify exports with deprecation note
  - [ ] Remove Beatport and TuneBat exports

## Phase 2: Remove Stubs and Update Docs

### 6. Remove Beatport/TuneBat Stubs
- [ ] Delete `src/edm/external/beatport.py`
- [ ] Delete `src/edm/external/tunebat.py`
- [ ] Update any imports that reference these modules
- [ ] Update `docs/architecture.md` to remove Beatport/TuneBat references

### 7. Unit Tests
- [ ] Create `tests/test_external/test_getsongbpm.py`
  - [ ] Test successful track search with mocked response
  - [ ] Test track not found returns None
  - [ ] Test rate limiting behavior
  - [ ] Test error handling (network errors, API errors)
  - [ ] Test caching behavior
  - [ ] Test missing API key handling
  - [ ] Mock HTTP responses for deterministic tests
- [ ] Update `tests/test_analysis/test_bpm.py`
  - [ ] Test new cascading strategy with GetSongBPM
  - [ ] Test fallback when GetSongBPM unavailable
  - [ ] Test Spotify deprecation warnings

### 8. Documentation Updates
- [ ] Update `docs/architecture.md`
  - [ ] Document GetSongBPM client
  - [ ] Update BPM analysis strategy diagram
  - [ ] Remove Beatport/TuneBat from external services list
  - [ ] Update "Placeholder / Unimplemented Features" section
- [ ] Update `docs/cli-reference.md`
  - [ ] Update BPM strategy explanation
  - [ ] Add GetSongBPM API key configuration
  - [ ] Document attribution requirement
- [ ] Update `docs/agent-guide.md`
  - [ ] Update external services section
- [ ] Update `README.md`
  - [ ] Update BPM detection strategy description
  - [ ] Add GetSongBPM API key setup instructions
  - [ ] Add attribution note

## Phase 3: Attribution and Deprecation

### 9. Implement Attribution Display
- [ ] Modify CLI table output to show attribution footer when GetSongBPM used
- [ ] Modify JSON output to include `attribution` field
- [ ] Add attribution note to --help text

### 10. Deprecate Spotify Client
- [ ] Add deprecation warning to `SpotifyClient.__init__()`
- [ ] Add deprecation notice to all Spotify-related docstrings
- [ ] Update `analyze_bpm()` to log deprecation warning when Spotify is used
- [ ] Document timeline for removal (target: v0.4.0)

### 11. Update CLI Help Text
- [ ] Update `src/cli/main.py` help text for `--offline` flag
- [ ] Update BPM strategy documentation in CLI
- [ ] Add note about GetSongBPM API key requirement

## Phase 4: Validation

### 12. Integration Tests
- [ ] Create integration test with real GetSongBPM API (marked as slow/optional)
- [ ] Test with known EDM tracks to verify BPM accuracy
- [ ] Verify caching works correctly

### 13. Accuracy Evaluation
- [ ] Run BPM accuracy evaluation with GetSongBPM as source
- [ ] Compare results against previous Spotify-based evaluation
- [ ] Document any accuracy differences

### 14. CI/CD Updates
- [ ] Add `GETSONGBPM_API_KEY` to GitHub Actions secrets (for integration tests)
- [ ] Ensure unit tests run with mocked responses (don't hit real API in CI)
- [ ] Update any environment variable documentation

## Final Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass (when API key available)
- [ ] Documentation is complete and accurate
- [ ] Attribution is displayed correctly
- [ ] Deprecation warnings are visible but not disruptive
- [ ] No breaking changes for users who don't use external APIs
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Beatport and TuneBat stubs removed
- [ ] spotipy can be optionally removed from dependencies
