# Change: Add Audio Caching Layer

## Why

Audio files are loaded multiple times during analysis - once for BPM detection, potentially again for structure analysis, and again if fallback algorithms are needed. `librosa.load()` is expensive (decoding + resampling). Caching loaded audio data eliminates redundant I/O and decoding, providing 20-40% speedup per file.

## What Changes

- Add `AudioCache` class to manage loaded audio data with LRU eviction
- Modify `compute_bpm_madmom()` and `compute_bpm_librosa()` to accept pre-loaded audio
- Add `load_audio()` function that caches results
- Integrate cache into analysis pipeline so audio is loaded once per file
- Add `--cache-size` CLI option to control memory usage (default: 10 files)
- Clear cache between batch operations to prevent memory bloat

## Impact

- **Affected specs**: analysis (new caching capability)
- **Affected code**:
  - New: `src/edm/io/audio.py` (AudioCache, load_audio)
  - Modified: `src/edm/analysis/bpm_detector.py` (accept pre-loaded audio)
  - Modified: `src/edm/analysis/bpm.py` (use cached loading)
  - Modified: `src/cli/commands/analyze.py` (cache lifecycle)
- **Performance**: 20-40% faster per-file analysis
- **Memory**: Configurable via cache size limit
- **Backward compatibility**: Full - caching is transparent
