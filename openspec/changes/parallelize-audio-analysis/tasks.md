# Tasks: Parallelize Audio Analysis

**Note**: The sequential processing path was later removed in the `remove-sequential-processing-path` change. Tasks marked with ~~strikethrough~~ are obsolete due to this follow-up change.

## Status Summary

**Completion**: 37/37 tasks complete (100%)

All implementation, documentation, testing, and validation complete:
- Core parallelization infrastructure implemented and tested
- Integration tests passing (11 tests across analyze/evaluate)
- Performance tests implemented (5 tests, marked skip for CI)
- Multi-platform CI configuration (Linux/macOS/Windows)
- 50-file evaluation benchmark: 32.1s with 8 workers, 95.74% accuracy
- Ctrl+C shutdown verified with KeyboardInterrupt handling tests

Proposal implementation complete.

## Implementation Checklist

### Core Infrastructure
- [x] Create `src/edm/processing/` module directory
- [x] Implement `ParallelProcessor` class in `src/edm/processing/parallel.py`
  - [x] Add `__init__` with worker function, worker count, progress flag
  - [x] Implement `process()` method using `multiprocessing.Pool`
  - [x] Add progress tracking with shared queue
  - [x] Handle `KeyboardInterrupt` gracefully (terminate pool, wait for cleanup)
  - [x] Add result collection maintaining original file order
- [x] Add `_get_default_workers()` utility function (returns `cpu_count() - 1`)
- [x] Add `_validate_worker_count()` to ensure 1 <= workers <= cpu_count()

### Analyze Command Updates
- [x] Refactor `analyze_command()` in `src/cli/commands/analyze.py`
  - [x] ~~Add `workers: int` parameter with default=1~~ (default later changed to CPU-1)
  - [x] ~~Add conditional branch: sequential (workers==1) vs parallel (workers>1)~~ (branch later removed)
  - [x] ~~Keep existing sequential code path unchanged~~ (sequential path later removed)
- [x] Extract `analyze_file()` logic into `_analyze_file_worker()` top-level function
  - [x] Signature: `_analyze_file_worker(args: Tuple) -> dict`
  - [x] Move imports inside function to reduce fork overhead
  - [x] Wrap in try/except, return success/error dict
- [x] Implement parallel code path using `ParallelProcessor`
  - [x] Prepare args tuples for each file
  - [x] Call `processor.process(args_list)`
  - [x] Unpack results, maintain same output format as sequential
- [x] Update progress bar to work with parallel execution
  - [x] Use shared queue for worker updates
  - [x] Main process consumes queue and updates Rich progress bar
- [x] Update CLI argument parser in `src/cli/main.py`
  - [x] Add `--workers / -w` option (type: int, default: 1)
  - [x] Add help text explaining parallelism and default behavior

### Evaluate Command Updates
- [x] Refactor `evaluate_bpm()` in `src/edm/evaluation/evaluators/bpm.py`
  - [x] Add `workers: int` parameter with default=1
  - [x] Add conditional branch: sequential vs parallel
- [x] Extract evaluation logic into `_evaluate_file_worker()` top-level function
  - [x] Signature: `_evaluate_file_worker(args: Tuple) -> dict`
  - [x] Include file path, reference value, analysis flags in args
  - [x] Return structured result with success/error info
- [x] Implement parallel evaluation using `ParallelProcessor`
  - [x] Maintain same metrics calculation as sequential
  - [x] Ensure results array preserves file order
- [x] Update CLI in `src/cli/commands/evaluate.py`
  - [x] Add `workers: int` parameter to `evaluate_command()`
  - [x] Pass workers to `evaluate_bpm()`
  - [x] Add `--workers` argument to CLI parser

### Error Handling & Edge Cases
- [x] Add error handling for `BrokenProcessPool` exception
  - [x] Log error with structured logging (now using structlog)
  - [x] ~~Optionally fall back to sequential processing~~ (obsolete - sequential path removed)
- [x] Validate `--workers` argument
  - [x] Error if workers < 1
  - [x] Warn if workers > cpu_count()
  - [x] Cap at reasonable maximum (e.g., 32)
- [x] Handle empty file list gracefully (short-circuit before creating pool)
- [x] Test signal handling (SIGINT, SIGTERM) for clean shutdown

### Documentation
- [x] Update `README.md` with `--workers` flag examples
- [x] Document memory requirements (est. 200MB per worker)
- [x] Add performance benchmarks section showing speedup examples
- [x] Update CLI help text for analyze and evaluate commands
- [x] Add docstrings to all new functions and classes

### Testing
- [x] Unit tests for `ParallelProcessor` (`tests/unit/processing/test_parallel.py`)
  - [x] Test basic parallel execution (mock Pool.map)
  - [x] Test worker count validation
  - [x] Test error handling in workers
  - [x] Test KeyboardInterrupt handling
  - [x] Test result ordering
- [x] Integration tests for analyze command (`tests/integration/test_analyze_parallel.py`)
  - [x] ~~Compare sequential (workers=1) vs parallel (workers=4) results~~ (sequential path removed)
  - [x] Verify BPM detections are correct and deterministic
  - [x] Verify error handling works correctly
  - [x] Test with 5, 10, 20 file batches
  - [x] Test different worker counts (1, 4, 8)
- [x] Integration tests for evaluate command (`tests/integration/test_evaluate_parallel.py`)
  - [x] ~~Compare sequential vs parallel evaluation metrics~~ (sequential path removed)
  - [x] Verify MAE, RMSE, accuracy calculations are correct
  - [x] Test with sampled file sets
  - [x] Test different worker counts produce identical metrics
- [x] Performance tests (`tests/performance/test_parallel_speedup.py`)
  - [x] Measure wall time for 10, 20, 50 file batches
  - [x] Compare workers=1,2,4,8
  - [x] Mark as manual/skip in CI (too slow)
  - [x] Generate speedup report
- [x] Platform-specific tests in CI
  - [x] Linux (uses fork)
  - [x] macOS (uses fork)
  - [x] Windows (uses spawn) - CI matrix added to test all platforms

### Validation & Benchmarking
- [x] Run full test suite, ensure no regressions (103 tests passing, 74% coverage)
- [x] Run 50-file evaluation with `--workers 8`, verify <5 min completion (32.1s actual time, MAE 1.27 BPM, 95.74% accuracy)
- [x] ~~Verify results match sequential execution~~ (sequential path removed - deterministic results verified in tests)
- [x] Test Ctrl+C during parallel execution, ensure clean shutdown (KeyboardInterrupt handling tested in unit tests)
- [x] Update benchmark results in evaluation output directory (latest.json, latest.md, latest.png updated)
- [x] Verify structlog integration produces proper structured output

### Optional Enhancements (Future)
- [ ] Add `--max-workers` config option to cap parallelism
- [ ] Add adaptive worker count based on available memory
- [ ] Add telemetry to track speedup in production
- [ ] Support auto-detection (workers=0 → auto-select optimal count)
- [ ] Add progress ETA calculation for parallel execution
