# Change: Parallelize Audio Analysis

## Why

Current audio analysis (both `analyze` and `evaluate` commands) processes files sequentially, one at a time. A 50-file BPM evaluation took ~20 minutes (avg 25s/file) because each file's madmom computation must complete before the next begins. This is severely inefficient on modern multi-core systems where CPU-bound analysis tasks can run concurrently. Users with large music libraries (hundreds or thousands of tracks) face impractically long wait times for batch analysis. Parallelization will reduce analysis time by 5-8x on typical 8-core systems, making the tool usable for production DJ workflows and large-scale evaluation.

## What Changes

- Add new `ParallelProcessor` class in `src/edm/processing/parallel.py` using multiprocessing.Pool
- Add `--workers N` CLI flag to both `analyze` and `evaluate` commands
- Refactor analyze and evaluate commands to support both sequential and parallel execution paths (later removed in `remove-sequential-processing-path`)
- Extract worker functions that can be pickled and executed in separate processes
- Implement progress tracking across parallel workers using shared queues
- Add graceful shutdown handling for KeyboardInterrupt and SIGTERM signals
- Validate worker count against system resources (CPU cores, memory)
- Update documentation with parallelization examples and performance benchmarks
- Migrate to structlog for structured logging throughout analysis pipeline

## Impact

- **Affected specs**: performance (new capability)
- **Affected code**:
  - New module: `src/edm/processing/parallel.py` (ParallelProcessor class)
  - Modified: `src/cli/commands/analyze.py` (add --workers flag, parallel path)
  - Modified: `src/cli/commands/evaluate.py` (add --workers flag)
  - Modified: `src/edm/evaluation/evaluators/bpm.py` (parallel evaluation)
  - Modified: `src/cli/main.py` (CLI argument parsing)
  - Modified: Multiple modules for structlog migration (`src/edm/analysis/`, `src/edm/external/`, etc.)
- **Performance**: 5-8x faster for batch operations on multi-core systems
- **Backward compatibility**: Initially full (workers=1 default), later changed to parallel-by-default in `remove-sequential-processing-path`
- **User experience**: Major improvement for large batch operations, optional complexity via --workers flag
- **Resource usage**: Increased memory usage (linear with worker count), better CPU utilization

## Design Decisions

### Why multiprocessing over threading?
BPM computation is CPU-bound (madmom, librosa, numpy operations). Python's GIL prevents true parallel CPU execution with threading. Multiprocessing bypasses GIL by using separate processes.

### Why default to CPU count - 1?
Leaves one core for system operations and UI responsiveness, preventing system lockup on heavily loaded machines. (Note: Initially defaulted to workers=1 for backward compatibility, later changed to CPU-1 in `remove-sequential-processing-path`)

### Why process-level vs file-level parallelism?
File-level parallelism is simpler, more effective, and aligns with the analysis workflow where each file is independent. No inter-file dependencies exist in BPM detection.

### Why remove the sequential code path?
Maintaining separate `_process_sequential()` and `_process_parallel()` functions added unnecessary complexity. The `ParallelProcessor` with `workers=1` is functionally identical to sequential processing. This decision was made in the follow-up `remove-sequential-processing-path` change.

## Impact Assessment

### Performance
- **Major improvement**: 5-8x speedup on typical 8-core systems
- **Memory**: Linear increase with worker count (each worker loads one file)
- **CPU**: Better utilization of available cores

### Compatibility
- **Initially backward compatible**: Sequential processing default (workers=1), later changed to parallel-by-default
- **No API changes**: Internal refactoring only
- **CLI addition**: New optional `--workers` flag
- **Breaking change (follow-up)**: Default behavior changed in `remove-sequential-processing-path` to use all CPUs by default

### Testing
- **Unit tests**: Test parallel worker pool mechanics
- **Integration tests**: Verify results match sequential execution
- **Performance tests**: Validate speedup claims

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory exhaustion with large files | High | Add `--workers` validation, document memory requirements |
| Platform-specific multiprocessing issues | Medium | Test on Linux, macOS, Windows; graceful fallback to sequential |
| Progress tracking complexity | Low | Use shared queue for worker updates |
| Zombie processes on error | Medium | Proper process pool cleanup, signal handling |

## Success Criteria

1. ✅ 50-file evaluation completes in ≤5 minutes on 8-core system (vs 20 min sequential)
2. ✅ Results are identical to sequential processing (bit-for-bit)
3. ✅ Ctrl+C cleanly terminates all workers without hanging
4. ✅ `--workers 1` produces identical behavior to pre-change sequential code (sequential path later removed)
5. ✅ Test coverage includes parallel execution paths
6. ✅ Structured logging with structlog integrated throughout analysis pipeline

## Alternatives Considered

### Alternative 1: asyncio with process pool
- **Pros**: More modern Python concurrency
- **Cons**: Adds complexity, no benefit over plain multiprocessing for this use case
- **Decision**: Rejected - unnecessary complexity

### Alternative 2: Ray or Dask
- **Pros**: Battle-tested distributed compute frameworks
- **Cons**: Heavy dependencies, overkill for local-only parallelism
- **Decision**: Rejected - prefer stdlib solutions first

### Alternative 3: GNU Parallel / shell-level parallelism
- **Pros**: Simple, language-agnostic
- **Cons**: Loses Python-level progress tracking, error handling, and integration
- **Decision**: Rejected - want tight integration with CLI
