# Tasks: Fix Cache Race Condition

## 1. Add Thread Lock
- [ ] 1.1 Import `threading` in `src/edm/io/audio.py`
- [ ] 1.2 Add `self._lock = threading.Lock()` to `AudioCache.__init__()`
- [ ] 1.3 Wrap `get()` method body with `with self._lock:`
- [ ] 1.4 Wrap `put()` method body with `with self._lock:`
- [ ] 1.5 Wrap `clear()` method body with `with self._lock:`
- [ ] 1.6 Wrap `stats()` method body with `with self._lock:`

## 2. Add Concurrency Tests
- [ ] 2.1 Create `tests/unit/test_audio_cache_threadsafe.py`
- [ ] 2.2 Add test for concurrent `put()` operations
- [ ] 2.3 Add test for concurrent `get()` operations
- [ ] 2.4 Add test for concurrent mixed operations
- [ ] 2.5 Add test for cache size consistency under load

## 3. Validation
- [ ] 3.1 Run existing audio cache tests: `pytest tests/unit/test_audio_cache.py`
- [ ] 3.2 Run new concurrency tests: `pytest tests/unit/test_audio_cache_threadsafe.py`
- [ ] 3.3 Test with multi-worker CLI: `edm analyze *.flac --workers 4`
- [ ] 3.4 Verify no deadlocks or crashes

## 4. Documentation
- [ ] 4.1 Add docstring note about thread safety to `AudioCache` class
- [ ] 4.2 Update CHANGELOG.md
