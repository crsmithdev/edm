# Tasks: Parallelize Audio Analysis

## Implementation Checklist

### Core Infrastructure
- [ ] Create `src/edm/processing/` module directory
- [ ] Implement `ParallelProcessor` class in `src/edm/processing/parallel.py`
  - [ ] Add `__init__` with worker function, worker count, progress flag
  - [ ] Implement `process()` method using `multiprocessing.Pool`
  - [ ] Add progress tracking with shared queue
  - [ ] Handle `KeyboardInterrupt` gracefully (terminate pool, wait for cleanup)
  - [ ] Add result collection maintaining original file order
- [ ] Add `_get_default_workers()` utility function (returns `cpu_count() - 1`)
- [ ] Add `_validate_worker_count()` to ensure 1 <= workers <= cpu_count()

### Analyze Command Updates
- [ ] Refactor `analyze_command()` in `src/cli/commands/analyze.py`
  - [ ] Add `workers: int` parameter with default=1
  - [ ] Add conditional branch: sequential (workers==1) vs parallel (workers>1)
  - [ ] Keep existing sequential code path unchanged
- [ ] Extract `analyze_file()` logic into `_analyze_file_worker()` top-level function
  - [ ] Signature: `_analyze_file_worker(args: Tuple) -> dict`
  - [ ] Move imports inside function to reduce fork overhead
  - [ ] Wrap in try/except, return success/error dict
- [ ] Implement parallel code path using `ParallelProcessor`
  - [ ] Prepare args tuples for each file
  - [ ] Call `processor.process(args_list)`
  - [ ] Unpack results, maintain same output format as sequential
- [ ] Update progress bar to work with parallel execution
  - [ ] Use shared queue for worker updates
  - [ ] Main process consumes queue and updates Rich progress bar
- [ ] Update CLI argument parser in `src/cli/main.py`
  - [ ] Add `--workers / -w` option (type: int, default: 1)
  - [ ] Add help text explaining parallelism and default behavior

### Evaluate Command Updates
- [ ] Refactor `evaluate_bpm()` in `src/edm/evaluation/evaluators/bpm.py`
  - [ ] Add `workers: int` parameter with default=1
  - [ ] Add conditional branch: sequential vs parallel
- [ ] Extract evaluation logic into `_evaluate_file_worker()` top-level function
  - [ ] Signature: `_evaluate_file_worker(args: Tuple) -> dict`
  - [ ] Include file path, reference value, analysis flags in args
  - [ ] Return structured result with success/error info
- [ ] Implement parallel evaluation using `ParallelProcessor`
  - [ ] Maintain same metrics calculation as sequential
  - [ ] Ensure results array preserves file order
- [ ] Update CLI in `src/cli/commands/evaluate.py`
  - [ ] Add `workers: int` parameter to `evaluate_command()`
  - [ ] Pass workers to `evaluate_bpm()`
  - [ ] Add `--workers` argument to CLI parser

### Error Handling & Edge Cases
- [ ] Add error handling for `BrokenProcessPool` exception
  - [ ] Log error with structured logging
  - [ ] Optionally fall back to sequential processing
- [ ] Validate `--workers` argument
  - [ ] Error if workers < 1
  - [ ] Warn if workers > cpu_count()
  - [ ] Cap at reasonable maximum (e.g., 32)
- [ ] Handle empty file list gracefully (short-circuit before creating pool)
- [ ] Test signal handling (SIGINT, SIGTERM) for clean shutdown

### Documentation
- [ ] Update `README.md` with `--workers` flag examples
- [ ] Document memory requirements (est. 200MB per worker)
- [ ] Add performance benchmarks section showing speedup examples
- [ ] Update CLI help text for analyze and evaluate commands
- [ ] Add docstrings to all new functions and classes

### Testing
- [ ] Unit tests for `ParallelProcessor` (`tests/unit/processing/test_parallel.py`)
  - [ ] Test basic parallel execution (mock Pool.map)
  - [ ] Test worker count validation
  - [ ] Test error handling in workers
  - [ ] Test KeyboardInterrupt handling
  - [ ] Test result ordering
- [ ] Integration tests for analyze command (`tests/integration/test_analyze_parallel.py`)
  - [ ] Compare sequential (workers=1) vs parallel (workers=4) results
  - [ ] Verify identical BPM detections
  - [ ] Verify identical error handling
  - [ ] Test with 5, 10, 20 file batches
- [ ] Integration tests for evaluate command (`tests/integration/test_evaluate_parallel.py`)
  - [ ] Compare sequential vs parallel evaluation metrics
  - [ ] Verify MAE, RMSE, accuracy are identical
  - [ ] Test with sampled file sets
- [ ] Performance tests (`tests/performance/test_parallel_speedup.py`)
  - [ ] Measure wall time for 10, 20, 50 file batches
  - [ ] Compare workers=1,2,4,8
  - [ ] Mark as manual/skip in CI (too slow)
  - [ ] Generate speedup report
- [ ] Platform-specific tests in CI
  - [ ] Linux (uses fork)
  - [ ] macOS (uses fork)
  - [ ] Windows (uses spawn) - ensure proper `__main__` guard

### Validation & Benchmarking
- [ ] Run 50-file evaluation with `--workers 8`, verify <5 min completion
- [ ] Verify results match sequential execution (diff JSON outputs)
- [ ] Test Ctrl+C during parallel execution, ensure clean shutdown
- [ ] Run full test suite, ensure no regressions
- [ ] Update benchmark results in evaluation output directory

### Optional Enhancements (Future)
- [ ] Add `--max-workers` config option to cap parallelism
- [ ] Add adaptive worker count based on available memory
- [ ] Add telemetry to track speedup in production
- [ ] Support auto-detection (workers=0 → auto-select optimal count)
- [ ] Add progress ETA calculation for parallel execution
