# Implementation Tasks

## Phase 1: Implement New Sources

### 1. Beatport Client Implementation
- [ ] Create `src/edm/external/beatport.py` web scraper
  - [ ] Implement track search by artist + title
  - [ ] Parse BPM from track page HTML
  - [ ] Parse key and genre information
  - [ ] Add rate limiting (respect robots.txt)
  - [ ] Add user agent and headers for respectful scraping
  - [ ] Handle HTTP errors and timeouts
  - [ ] Return `BeatportTrackInfo` dataclass

### 2. TuneBat Client Implementation
- [ ] Create `src/edm/external/tunebat.py` web scraper
  - [ ] Implement track search by artist + title
  - [ ] Parse BPM from results page
  - [ ] Parse key and camelot key information
  - [ ] Add rate limiting (respect robots.txt)
  - [ ] Add user agent and headers for respectful scraping
  - [ ] Handle HTTP errors and timeouts
  - [ ] Return `TuneBatTrackInfo` dataclass

### 3. Add Dependencies
- [ ] Add `beautifulsoup4` to pyproject.toml
- [ ] Add `requests` to pyproject.toml
- [ ] Add `lxml` parser for beautifulsoup (optional but faster)
- [ ] Update `uv.lock` with new dependencies

### 4. Update BPM Analysis Strategy
- [ ] Modify `src/edm/analysis/bpm.py`
  - [ ] Add `_try_beatport()` helper function
  - [ ] Add `_try_tunebat()` helper function
  - [ ] Update `analyze_bpm()` to use new strategy: metadata → beatport → tunebat → computed
  - [ ] Add deprecation warning when Spotify is used
  - [ ] Keep `_try_spotify()` for backward compatibility

### 5. Update Configuration
- [ ] Modify `src/edm/config.py`
  - [ ] Add Beatport configuration section (rate_limit, cache_ttl, user_agent)
  - [ ] Add TuneBat configuration section (rate_limit, cache_ttl, user_agent)
  - [ ] Add deprecation notice to Spotify config section
  - [ ] Update default BPM lookup strategy to `["metadata", "beatport", "tunebat", "computed"]`
  - [ ] Add configuration option for source preference order

## Phase 2: Testing and Documentation

### 6. Unit Tests
- [ ] Create `tests/test_external/test_beatport.py`
  - [ ] Test successful track search
  - [ ] Test rate limiting behavior
  - [ ] Test error handling (network errors, parsing errors)
  - [ ] Test timeout handling
  - [ ] Mock HTTP responses for deterministic tests
- [ ] Create `tests/test_external/test_tunebat.py`
  - [ ] Test successful track search
  - [ ] Test rate limiting behavior
  - [ ] Test error handling
  - [ ] Test timeout handling
  - [ ] Mock HTTP responses for deterministic tests
- [ ] Update `tests/test_analysis/test_bpm.py`
  - [ ] Test new cascading strategy
  - [ ] Test Beatport fallback when metadata missing
  - [ ] Test TuneBat fallback when Beatport fails
  - [ ] Test Spotify deprecation warnings

### 7. Integration Tests
- [ ] Create integration tests with real web scraping (marked as slow)
- [ ] Test Beatport scraping with known tracks
- [ ] Test TuneBat scraping with known tracks
- [ ] Verify caching works across sources

### 8. Documentation Updates
- [ ] Update `docs/architecture.md`
  - [ ] Document new Beatport and TuneBat clients
  - [ ] Update BPM analysis strategy diagram
  - [ ] Document web scraping approach and limitations
- [ ] Update `docs/cli-reference.md`
  - [ ] Update BPM strategy explanation
  - [ ] Remove references to Spotify as primary source
  - [ ] Add migration guide for users
- [ ] Create migration guide
  - [ ] Document how to update configurations
  - [ ] Explain new BPM strategy
  - [ ] Note backward compatibility period
- [ ] Update README.md
  - [ ] Update BPM detection strategy description
  - [ ] Update feature list (Beatport, TuneBat instead of Spotify)

## Phase 3: Deprecation and Cleanup

### 9. Deprecate Spotify Client
- [ ] Add deprecation warning to `SpotifyClient.__init__()`
- [ ] Add deprecation notice to all Spotify-related docstrings
- [ ] Update `analyze_bpm()` to log deprecation warning when Spotify is used
- [ ] Document timeline for removal (target: v0.4.0)

### 10. Update CLI Help Text
- [ ] Update `src/cli/main.py` help text for `--offline` flag
- [ ] Update BPM strategy documentation in CLI
- [ ] Add note about Spotify deprecation

### 11. Validation and Accuracy Testing
- [ ] Run accuracy evaluation with new sources
- [ ] Compare results against Spotify-based evaluation
- [ ] Verify BPM accuracy meets or exceeds previous results
- [ ] Document any accuracy changes in evaluation results

### 12. CI/CD Updates
- [ ] Update GitHub Actions workflows if needed
- [ ] Add environment variables for optional rate limit configuration
- [ ] Ensure tests run with mocked HTTP responses (don't hit real services in CI)

## Final Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Documentation is complete and accurate
- [ ] Migration guide is clear and tested
- [ ] Deprecation warnings are visible but not disruptive
- [ ] No breaking changes for users who don't configure Spotify
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Accuracy evaluation shows comparable or better results
