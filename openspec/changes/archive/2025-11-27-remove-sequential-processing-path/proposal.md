# Change: Remove Sequential Processing Path

## Why

The codebase currently maintains two separate code paths for file processing: `_process_sequential()` and `_process_parallel()`. This duplication exists for historical reasons (sequential was default before parallelization was added), but is unnecessary complexity. The `ParallelProcessor` with `workers=1` is functionally identical to sequential processing - it still processes files one at a time, just uses a process pool with a single worker. The only differences are cosmetic (progress bar styling) and not worth maintaining separate implementations.

Modern systems have multiple cores by default. Defaulting to sequential processing (workers=1) wastes available CPU resources and forces users to discover and enable the `--workers` flag manually. Most users never realize they could be running 5-8x faster.

## What Changes

- Remove `_process_sequential()` function from `src/cli/commands/analyze.py`
- Always use `_process_parallel()` for all batch processing
- Change default workers from `1` to `get_default_workers()` (CPU count - 1)
- Unify progress bar styling (always show bar + completion counts)
- Update CLI help text to reflect parallel-by-default behavior
- Remove conditional branch that selects between sequential/parallel paths

## Impact

- **Affected specs**: `cli`
- **Affected code**:
  - Modified: `src/cli/commands/analyze.py` (remove `_process_sequential`, change default)
  - Modified: `src/cli/main.py` (change default workers value)
  - Removed: ~40 lines of duplicate sequential processing code
- **Performance**: Users get 5-8x speedup by default without manual flag configuration
- **User experience**: BREAKING - different default behavior, but strictly better performance
- **Backward compatibility**: Users can still set `--workers 1` for single-threaded behavior if needed
