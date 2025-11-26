## 1. Core Implementation

- [x] 1.1 Create `src/edm/io/audio.py` with `AudioCache` class using `OrderedDict` for LRU
- [x] 1.2 Implement `load_audio(filepath, sr=None)` function returning `(y, sr)` tuple
- [x] 1.3 Add `get_audio_cache()`, `set_cache_size()`, `clear_audio_cache()` global functions
- [x] 1.4 Add `clear()` method and `stats()` method to AudioCache

## 2. Integration

- [x] 2.1 Modify `compute_bpm_librosa()` to accept optional `audio` parameter
- [x] 2.2 Note: madmom loads its own audio internally, caching benefits librosa fallback
- [x] 2.3 Update `_try_compute()` in bpm.py to use cached audio loading
- [x] 2.4 Integrate cache clearing into analyze command (finally block)

## 3. CLI

- [x] 3.1 Add `--cache-size` option to analyze command (default: 10)
- [x] 3.2 Cache configured via `set_cache_size()` before analysis starts

## 4. Testing

- [x] 4.1 Unit test AudioCache LRU behavior (test_cache_lru_eviction)
- [x] 4.2 Unit test load_audio with caching (test_load_audio_caches_result)
- [x] 4.3 Unit test cache disabled when size=0 (test_cache_disabled_when_size_zero)
- [x] 4.4 Unit test cache stats tracking (test_cache_stats)
- [x] 4.5 All 72 tests pass (59 existing + 13 new)
