# Design: Parallelize Audio Analysis

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLI Commands                         │
│            (analyze.py, evaluate.py)                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ├─ Sequential Mode (--workers 1)
                   │  └─> Direct function calls
                   │
                   └─ Parallel Mode (--workers N)
                      └─> ParallelProcessor
                          ├─ Process Pool (multiprocessing.Pool)
                          ├─ Task Queue
                          ├─ Result Collection
                          └─ Progress Tracking (shared queue)
```

## Component Design

### 1. ParallelProcessor Class

**Location**: `src/edm/processing/parallel.py` (new module)

```python
class ParallelProcessor:
    """Parallel file processor with progress tracking."""

    def __init__(self,
                 worker_func: Callable,
                 num_workers: int,
                 show_progress: bool = True):
        """
        Args:
            worker_func: Function to execute per file (must be picklable)
            num_workers: Number of parallel workers
            show_progress: Display progress bar
        """

    def process(self, items: List[Any]) -> List[Result]:
        """Process items in parallel with progress tracking."""
```

**Key Features**:
- Uses `multiprocessing.Pool` with context manager for automatic cleanup
- Implements graceful shutdown on `KeyboardInterrupt`
- Collects results in original order via indexed queue
- Reports progress via `multiprocessing.Manager().Queue()`

### 2. Worker Functions

Worker functions must be:
- **Top-level functions** (not lambdas/closures) - required for pickling
- **Pure**: No global state mutations
- **Error-wrapped**: Catch exceptions, return Result objects
- **Minimal imports**: Import expensive libraries inside function

**Example Worker Signature**:
```python
def _analyze_file_worker(args: Tuple[Path, bool, bool, bool, bool]) -> dict:
    """Worker function for parallel file analysis.

    Args:
        args: (filepath, run_bpm, run_structure, offline, ignore_metadata)

    Returns:
        Dict with results or error information
    """
    filepath, run_bpm, run_structure, offline, ignore_metadata = args
    try:
        # Import heavy deps inside worker to avoid fork overhead
        from edm.analysis.bpm import analyze_bpm
        from edm.analysis.structure import analyze_structure

        result = {}
        # ... analysis logic ...
        return {"success": True, "result": result, "file": str(filepath)}
    except Exception as e:
        return {"success": False, "error": str(e), "file": str(filepath)}
```

### 3. Progress Tracking

**Challenge**: Rich progress bars don't work across process boundaries
**Solution**: Use a shared queue for worker updates, main process updates UI

```python
# In worker
progress_queue.put({"file": filepath, "status": "complete"})

# In main process
with Progress() as progress:
    task = progress.add_task("Processing...", total=len(items))
    while active_workers:
        update = progress_queue.get(timeout=0.1)
        progress.update(task, advance=1)
```

### 4. CLI Integration

**Changes to `analyze.py`**:
```python
def analyze_command(..., workers: int = 1, ...):
    if workers == 1:
        # Sequential path (existing code)
        results = _analyze_sequential(audio_files, ...)
    else:
        # Parallel path (new code)
        processor = ParallelProcessor(_analyze_file_worker, workers)
        results = processor.process(audio_files)
```

**Changes to `evaluate.py`**:
Similar pattern for `evaluate_bpm()` function.

## Data Flow

### Sequential (Existing)
```
Files → for file in files → analyze_file() → results.append() → results
```

### Parallel (New)
```
Files → Pool.map(worker_func, files) → [Result] → aggregate → results
        ↓
    progress_queue → main process → UI updates
```

## Error Handling Strategy

### Worker-Level Errors
- Catch all exceptions in worker function
- Return structured error dict: `{"success": False, "error": msg}`
- Continue processing remaining files

### Process-Level Errors
- Wrap pool operations in try/finally for cleanup
- Handle `KeyboardInterrupt` specifically - terminate pool gracefully
- Handle `BrokenProcessPool` - fall back to sequential or re-raise

### Resource Exhaustion
- Validate `--workers` against system CPU count
- Add `--max-workers` config option (default: `cpu_count() - 1`)
- Document memory requirements in error messages

## Platform Considerations

### macOS/Linux
- Use `fork` start method (default) - fast, shares memory initially
- Watch for issues with shared file descriptors

### Windows
- Use `spawn` start method (default) - slower, no memory sharing
- Requires `if __name__ == "__main__"` guard (already present in CLI)

### Testing
- Mock `multiprocessing.Pool` for deterministic tests
- Use `workers=1` for regression tests (deterministic, easier debugging)
- Add platform-specific integration tests in CI

## Memory Management

### Per-Worker Memory
Each worker loads:
- Audio file into memory (~50-100 MB for 5min FLAC)
- madmom model (~100 MB)
- Librosa/numpy buffers (~50 MB)

**Total**: ~200 MB per worker

### Safe Worker Counts
- 8 GB RAM: max 4-6 workers
- 16 GB RAM: max 10-12 workers
- 32 GB RAM: limited by CPU count

**Default**: `min(cpu_count() - 1, max_safe_workers_for_ram())`

## Performance Model

### Expected Speedup
```
Speedup = min(N_workers, N_files) / (1 + overhead_factor)

where:
  overhead_factor ≈ 0.1-0.2 (process spawn, IPC, result collection)
```

**Example**: 50 files, 8 workers
- Ideal speedup: 8x
- Realistic speedup: ~6-7x (accounting for overhead)

### Bottlenecks
1. **Process spawn time**: ~50-100ms per worker (amortized over many files)
2. **Result collection**: Queue overhead increases with worker count
3. **I/O contention**: If files on HDD (not SSD), parallel reads may slow down

## Testing Strategy

### Unit Tests
- `test_parallel_processor.py`: Test worker pool mechanics
- Mock `Pool.map` to verify correct function/args passed
- Test error handling in workers

### Integration Tests
- `test_parallel_analyze.py`: Compare sequential vs parallel results
- Verify identical outputs (BPM, confidence, metadata)
- Test with `--workers 1,2,4,8`

### Performance Tests
- `test_parallel_performance.py`: Benchmark speedup
- Skip in CI, run manually for validation
- Measure wall time for 10,20,50 file batches

## Configuration

### CLI Arguments
```bash
edm analyze files... --workers N    # Number of parallel workers
edm evaluate bpm ... --workers N    # Same for evaluate
```

### Config File
```toml
[analysis]
max_workers = 8        # System-wide cap
default_workers = -1   # -1 = auto-detect
```

### Environment Variables
```bash
EDM_MAX_WORKERS=8      # Override for resource-constrained envs
```

## Migration Path

### Phase 1: Implement Core (This Change)
- Add `ParallelProcessor` class
- Refactor analyze/evaluate to support `--workers`
- Default to `workers=1` (sequential, no behavior change)

### Phase 2: Enable by Default (Future)
- Change default to `workers=0` (auto-detect optimal)
- Monitor for platform-specific issues
- Add telemetry for speedup measurements

### Phase 3: Advanced Features (Future)
- GPU acceleration for madmom
- Adaptive worker count based on memory pressure
- Distributed processing across network nodes

## Open Questions

1. **Should we batch small files to reduce process spawn overhead?**
   - Decision: No, premature optimization. Revisit if profiling shows spawn time dominates.

2. **Should we support mixed CPU/GPU workers?**
   - Decision: Out of scope. Focus on CPU-only for initial implementation.

3. **How to handle SIGTERM vs SIGKILL?**
   - Decision: Catch SIGTERM for graceful shutdown, document that SIGKILL may leave zombies.

4. **Should progress show per-file updates or batch updates?**
   - Decision: Per-file for responsiveness. Queue overhead is negligible for 50-1000 files.
