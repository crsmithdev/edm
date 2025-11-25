# Change: Complete structlog migration

## Why

The structlog migration is partially complete. Most modules use structlog, but 5 files still use stdlib `logging`. Additionally, existing structlog calls use snake_case event names (e.g., `"bpm_analysis_started"`) instead of human-readable phrases (e.g., `"BPM analysis started"`). This inconsistency makes logs harder to read and grep.

## Current State

**Already migrated to structlog (12 files):**
- `src/edm/analysis/bpm.py`
- `src/edm/analysis/bpm_detector.py`
- `src/edm/analysis/structure.py`
- `src/edm/config.py`
- `src/edm/evaluation/common.py`
- `src/edm/evaluation/evaluators/bpm.py`
- `src/edm/evaluation/reference.py`
- `src/edm/external/spotify.py`
- `src/edm/io/metadata.py`
- `src/edm/logging.py`
- `src/cli/commands/analyze.py`
- `src/cli/main.py`

**Still using stdlib logging (5 files):**
- `src/edm/external/beatport.py`
- `src/edm/external/tunebat.py`
- `src/edm/features/spectral.py`
- `src/edm/features/temporal.py`
- `src/edm/models/base.py`

Note: These 5 files are stub code that may be removed by the `remove-stub-code` proposal. If removed, no stdlib migration needed.

## What Changes

1. **Migrate remaining stdlib files** (if not removed by `remove-stub-code`):
   - Replace `import logging` with `import structlog`
   - Replace `logging.getLogger(__name__)` with `structlog.get_logger(__name__)`
   - Convert string-formatted messages to structured key-value pairs

2. **Update event names to human-readable phrases**:
   - Change `"bpm_analysis_started"` → `"BPM analysis started"`
   - Change `"file_analyzed"` → `"File analyzed"`
   - Change `"spotify_lookup_failed"` → `"Spotify lookup failed"`
   - Apply consistently across all structlog calls

3. **Standardize context field naming**:
   - Use consistent field names: `filepath`, `filename`, `elapsed_time`, `error`
   - Round numeric values appropriately

## Impact

- **Affected code**: All files listed above
- **No breaking changes**: Internal logging only
- **Developer experience**: More readable logs, consistent formatting
