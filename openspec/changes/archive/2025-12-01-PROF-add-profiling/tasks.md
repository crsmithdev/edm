## 1. Core Profiling Infrastructure

- [x] 1.1 Create `src/edm/profiling/__init__.py` module
- [x] 1.2 Implement `ProfilerBase` abstract class with start(), stop(), report() methods
- [x] 1.3 Add profiling context manager for with-statement usage
- [x] 1.4 Create `ProfileResult` dataclass for storing profiling data

## 2. CPU Profiling Implementation

- [x] 2.1 Implement `CPUProfiler` class wrapping cProfile
- [x] 2.2 Add function to export cProfile stats to JSON format
- [ ] 2.3 Integrate py-spy for flamegraph generation
- [ ] 2.4 Add `--flamegraph` flag support in CPUProfiler

## 3. Memory Profiling Implementation

- [ ] 3.1 Implement `MemoryProfiler` class wrapping tracemalloc
- [ ] 3.2 Add snapshot capture and comparison functionality
- [ ] 3.3 Create memory allocation report formatter
- [ ] 3.4 Add peak memory tracking

## 4. Baseline Management

- [ ] 4.1 Create `benchmarks/baselines/` directory structure
- [ ] 4.2 Implement `BaselineStore` class for JSON read/write
- [ ] 4.3 Add `save_baseline(name, profile_result)` function
- [ ] 4.4 Add `compare_baseline(name, current_result)` function with threshold checking
- [ ] 4.5 Include git commit hash and timestamp in baseline metadata

## 5. CLI Integration

- [ ] 5.1 Add `--profile` flag to `edm analyze` command
- [ ] 5.2 Add `--profile` flag to `edm evaluate` command
- [ ] 5.3 Support multiple profile types: `--profile cpu,memory`
- [ ] 5.4 Add `--save-baseline <name>` flag
- [ ] 5.5 Add `--compare-baseline <name>` flag
- [ ] 5.6 Add `--format` flag for output format (console/json/html)

## 6. Report Generation

- [ ] 6.1 Implement console reporter with Rich table formatting
- [ ] 6.2 Implement JSON reporter for machine-readable output
- [ ] 6.3 Implement HTML reporter with embedded flamegraphs
- [ ] 6.4 Add top-N function highlighting in reports

## 7. Profiling Decorator

- [ ] 7.1 Implement `@profile_function` decorator
- [ ] 7.2 Add `EDM_PROFILE` environment variable check
- [ ] 7.3 Ensure zero overhead when profiling disabled
- [ ] 7.4 Store decorated function results in thread-safe registry

## 8. pytest Integration

- [ ] 8.1 Add `@pytest.mark.benchmark` marker to pyproject.toml
- [ ] 8.2 Create `tests/performance/` directory
- [ ] 8.3 Implement pytest fixture for baseline comparison
- [ ] 8.4 Add sample benchmark test for BPM analysis
- [ ] 8.5 Configure pytest to skip benchmarks by default

## 9. Dependencies

- [ ] 9.1 Add `py-spy` to dev dependencies in pyproject.toml
- [ ] 9.2 Add type stubs for cProfile if needed
- [ ] 9.3 Update mypy config to ignore py-spy if no stubs available

## 10. Documentation

- [ ] 10.1 Create `docs/profiling.md` with usage guide
- [ ] 10.2 Add profiling examples to docs
- [ ] 10.3 Document baseline comparison workflow
- [ ] 10.4 Add profiling section to CLI reference docs

## 11. Tests

- [x] 11.1 Unit tests for CPUProfiler class
- [ ] 11.2 Unit tests for MemoryProfiler class
- [ ] 11.3 Unit tests for BaselineStore
- [ ] 11.4 Integration test: profile analyze command end-to-end
- [ ] 11.5 Integration test: baseline save and compare
- [ ] 11.6 Test profiling decorator with and without EDM_PROFILE set

## 12. Validation

- [ ] 12.1 Run profiling on real audio files to verify overhead is minimal
- [ ] 12.2 Verify flamegraphs render correctly in browser
- [ ] 12.3 Validate baseline comparison threshold detection
- [ ] 12.4 Check that profiling has zero impact when disabled
