# Change: Remove Stub Code

## Why

~900 LOC of stub/placeholder implementations exist that return hardcoded or empty values. This creates confusion (code appears complete but isn't), bloats the codebase, and may mislead users who expect functionality that doesn't exist. Removing unused stubs follows YAGNI and keeps the codebase honest.

## What Changes

### Remove unused stub modules (not called anywhere):
- Remove `src/edm/features/spectral.py` (returns zero arrays)
- Remove `src/edm/features/temporal.py` (returns zero arrays)
- Remove `src/edm/features/` directory entirely
- Remove `src/edm/external/beatport.py` (stub with NotImplementedError)
- Remove `src/edm/external/tunebat.py` (stub with NotImplementedError)
- Remove `src/edm/models/base.py` (stub with NotImplementedError)
- Remove `src/edm/models/` directory entirely
- Update `src/edm/external/__init__.py` to remove beatport/tunebat exports

### Keep structure analysis stub but mark unimplemented:
- Keep `src/edm/analysis/structure.py` (called by CLI)
- Modify `analyze_structure()` to return empty result with `implemented=False` flag
- Update CLI to display "not implemented" message when structure analysis requested
- Keep `--analysis structure` option for forward compatibility

## Impact

- **Affected specs**: None (removing non-functional code)
- **Affected code**:
  - Removed: `src/edm/features/` directory
  - Removed: `src/edm/external/beatport.py`
  - Removed: `src/edm/external/tunebat.py`
  - Removed: `src/edm/models/` directory
  - Modified: `src/edm/analysis/structure.py` (return unimplemented marker)
  - Modified: `src/cli/commands/analyze.py` (handle unimplemented structure)
  - Modified: `src/edm/external/__init__.py`
- **LOC reduction**: ~700 lines removed
- **No breaking changes**: CLI options preserved, structure just shows "not implemented"
