# Tasks: Fix Cache Race Condition

## 1. Add Thread Lock
- [x] 1.1 Import `threading` in `src/edm/io/audio.py`
- [x] 1.2 Add `self._lock = threading.Lock()` to `AudioCache.__init__()`
- [x] 1.3 Wrap `get()` method body with `with self._lock:`
- [x] 1.4 Wrap `put()` method body with `with self._lock:`
- [x] 1.5 Wrap `clear()` method body with `with self._lock:`
- [x] 1.6 Wrap `stats()` method body with `with self._lock:`

## 2. Add Concurrency Tests
- [x] 2.1 Create `tests/unit/test_audio_cache_threadsafe.py`
- [x] 2.2 Add test for concurrent `put()` operations
- [x] 2.3 Add test for concurrent `get()` operations
- [x] 2.4 Add test for concurrent mixed operations
- [x] 2.5 Add test for cache size consistency under load

## 3. Validation
- [x] 3.1 Run existing audio cache tests: `pytest tests/unit/test_audio_cache.py`
- [x] 3.2 Run new concurrency tests: `pytest tests/unit/test_audio_cache_threadsafe.py`
- [x] 3.3 Test with multi-worker CLI: `edm analyze *.flac --workers 4`
- [x] 3.4 Verify no deadlocks or crashes

## 4. Documentation
- [x] 4.1 Add docstring note about thread safety to `AudioCache` class
- [x] 4.2 Update CHANGELOG.md (in commit message)
