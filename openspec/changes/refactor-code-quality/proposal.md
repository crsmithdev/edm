# Change: Code Quality Refactoring

## Why

Several code quality issues reduce maintainability and type safety:
1. **LRU cache on instance methods** in SpotifyClient creates shared class-level caches that persist incorrectly
2. **Lazy imports in hot paths** hurt discoverability and add per-call overhead
3. **String literals for BPM sources** instead of enums reduce type safety and IDE support

## What Changes

### 1. Fix SpotifyClient Caching
- Replace `@lru_cache` on instance methods with proper per-instance caching
- Use `cachetools.TTLCache` or manual dict-based cache bound to instance

### 2. Remove Lazy Imports
- Move imports from inside `_try_metadata()`, `_try_spotify()`, `_try_compute()` to module level
- Move `from edm.exceptions import AnalysisError` to top of files

### 3. Add BPMSource Enum
- Create `BPMSource` enum with `METADATA`, `SPOTIFY`, `COMPUTED` values
- Update `BPMResult.source` type from `Literal[...]` to `BPMSource`
- Update all references to use enum values

## Impact

- **Affected specs**: None (internal refactoring)
- **Affected code**:
  - Modified: `src/edm/external/spotify.py` (cache fix)
  - Modified: `src/edm/analysis/bpm.py` (imports, enum)
  - Modified: `src/edm/analysis/bpm_detector.py` (imports)
  - New: `src/edm/types.py` or add enum to `src/edm/analysis/bpm.py`
- **Type safety**: Improved with enum usage
- **Performance**: Slight improvement from removing lazy imports
- **Backward compatibility**: Full - enum values match existing strings
