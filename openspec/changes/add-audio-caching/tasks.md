## 1. Core Implementation

- [ ] 1.1 Create `src/edm/io/audio.py` with `AudioCache` class using `functools.lru_cache`
- [ ] 1.2 Implement `load_audio(filepath, sr=None)` function returning `(y, sr)` tuple
- [ ] 1.3 Add cache size configuration to `EDMConfig`
- [ ] 1.4 Add `clear_cache()` function for memory management

## 2. Integration

- [ ] 2.1 Modify `compute_bpm_librosa()` to accept optional `(y, sr)` parameter
- [ ] 2.2 Modify `compute_bpm_madmom()` to accept optional pre-loaded audio (if madmom supports it)
- [ ] 2.3 Update `_try_compute()` in bpm.py to use cached audio loading
- [ ] 2.4 Integrate cache clearing into analyze command batch loop

## 3. CLI

- [ ] 3.1 Add `--cache-size` option to analyze command (default: 10)
- [ ] 3.2 Document cache behavior in CLI help text

## 4. Testing

- [ ] 4.1 Unit test AudioCache LRU behavior
- [ ] 4.2 Unit test load_audio with various formats
- [ ] 4.3 Integration test: verify analysis results unchanged with caching
- [ ] 4.4 Performance test: measure speedup on repeated analysis
