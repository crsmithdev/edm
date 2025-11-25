## 1. Fix SpotifyClient Caching

- [ ] 1.1 Add `cachetools` to dependencies or implement manual cache
- [ ] 1.2 Replace `@lru_cache` on `search_track()` with instance-level `TTLCache`
- [ ] 1.3 Replace `@lru_cache` on `get_audio_features()` with instance-level cache
- [ ] 1.4 Add `clear_cache()` method to SpotifyClient
- [ ] 1.5 Update tests to verify cache is per-instance

## 2. Remove Lazy Imports

- [ ] 2.1 Move `from edm.io.metadata import read_metadata` to top of bpm.py
- [ ] 2.2 Move `from edm.external.spotify import SpotifyClient` to top of bpm.py
- [ ] 2.3 Move `from edm.analysis.bpm_detector import compute_bpm` to top of bpm.py
- [ ] 2.4 Move `from edm.exceptions import AnalysisError` to top of bpm.py
- [ ] 2.5 Move exception imports to top of bpm_detector.py
- [ ] 2.6 Verify no circular import issues after changes

## 3. Add BPMSource Enum

- [ ] 3.1 Create `BPMSource` enum in `src/edm/analysis/bpm.py`
- [ ] 3.2 Update `BPMResult.source` type annotation to use `BPMSource`
- [ ] 3.3 Update `_try_metadata()` to return `BPMSource.METADATA`
- [ ] 3.4 Update `_try_spotify()` to return `BPMSource.SPOTIFY`
- [ ] 3.5 Update `_try_compute()` to return `BPMSource.COMPUTED`
- [ ] 3.6 Add `__str__` method to enum for backward-compatible string output
- [ ] 3.7 Update CLI output code to handle enum values
- [ ] 3.8 Update tests to use enum values

## 4. Validation

- [ ] 4.1 Run full test suite
- [ ] 4.2 Run mypy type checking
- [ ] 4.3 Run ruff linting
- [ ] 4.4 Manual test CLI to verify output unchanged
