## ADDED Requirements

### Requirement: CPU Profiling with cProfile

The system SHALL provide CPU profiling using Python's cProfile module for identifying performance bottlenecks in function execution.

#### Scenario: Profile CPU usage during analysis
- **WHEN** user runs `edm analyze track.mp3 --profile cpu`
- **THEN** system collects CPU profiling data using cProfile
- **AND** saves profile statistics to `profiles/cpu_<timestamp>.prof`
- **AND** displays top 20 functions by cumulative time

#### Scenario: Generate flamegraph from CPU profile
- **WHEN** user runs `edm analyze track.mp3 --profile cpu --flamegraph`
- **THEN** system generates SVG flamegraph using py-spy
- **AND** saves to `profiles/cpu_<timestamp>.svg`
- **AND** displays path to flamegraph file

### Requirement: Memory Profiling with tracemalloc

The system SHALL provide memory profiling using Python's tracemalloc module for tracking memory allocations and identifying memory leaks.

#### Scenario: Profile memory usage during analysis
- **WHEN** user runs `edm analyze track.mp3 --profile memory`
- **THEN** system tracks memory allocations using tracemalloc
- **AND** saves memory snapshot to `profiles/memory_<timestamp>.json`
- **AND** displays top 10 allocation sites by size

#### Scenario: Compare memory usage between runs
- **WHEN** user runs profiling twice with different configurations
- **THEN** system SHALL allow comparing memory snapshots
- **AND** report difference in peak memory usage

### Requirement: Baseline Storage and Comparison

The system SHALL store profiling baselines as JSON files and provide comparison against previous runs for regression detection.

#### Scenario: Save profiling baseline
- **WHEN** user runs `edm analyze track.mp3 --profile cpu --save-baseline main`
- **THEN** system saves CPU and wall-clock time to `benchmarks/baselines/main.json`
- **AND** includes git commit hash, timestamp, and system info

#### Scenario: Compare against baseline
- **WHEN** user runs `edm analyze track.mp3 --profile cpu --compare-baseline main`
- **THEN** system loads baseline from `benchmarks/baselines/main.json`
- **AND** displays percentage change in CPU time and memory usage
- **AND** warns if current run is >20% slower than baseline

#### Scenario: Baseline format
- **WHEN** baseline file is saved
- **THEN** it MUST contain commit hash, timestamp, profile name, cpu_time, wall_time, and peak_memory_mb fields

### Requirement: Profiling Decorator for Functions

The system SHALL provide a decorator for profiling specific functions without requiring CLI flags.

#### Scenario: Decorate function for automatic profiling
- **WHEN** developer adds `@profile_function` decorator to a function
- **THEN** function SHALL automatically collect timing and memory data when called
- **AND** results SHALL be accessible via profiling context manager

#### Scenario: Conditional profiling based on environment
- **WHEN** `EDM_PROFILE=1` environment variable is set
- **THEN** all decorated functions SHALL be profiled
- **WHEN** environment variable is not set
- **THEN** decorator SHALL have zero overhead (no-op)

### Requirement: Profile Report Generation

The system SHALL generate human-readable profiling reports in multiple formats.

#### Scenario: Generate console report
- **WHEN** profiling completes
- **THEN** system displays summary table with function name, calls, total time, and cumulative time
- **AND** highlights top 5 slowest functions

#### Scenario: Generate JSON report
- **WHEN** user runs `edm analyze track.mp3 --profile cpu --format json`
- **THEN** system saves profiling data as JSON to `profiles/report_<timestamp>.json`
- **AND** JSON SHALL include all function statistics in machine-readable format

#### Scenario: Generate HTML report
- **WHEN** user runs `edm analyze track.mp3 --profile cpu --format html`
- **THEN** system generates HTML report with interactive tables
- **AND** includes flamegraph visualization embedded in HTML

### Requirement: Performance Regression Tests

The system SHALL support pytest integration for automated performance regression testing.

#### Scenario: Mark test as performance benchmark
- **WHEN** test is marked with `@pytest.mark.benchmark`
- **THEN** test SHALL run only when `pytest -m benchmark` is invoked
- **AND** test execution SHALL be profiled and compared against baseline

#### Scenario: Fail test on significant regression
- **WHEN** performance test runs slower than 120% of baseline
- **THEN** test SHALL fail with clear regression message
- **AND** display current vs baseline timing

### Requirement: Low Overhead When Disabled

The system SHALL ensure profiling infrastructure has zero performance impact when not actively profiling.

#### Scenario: Analyze without profiling flag
- **WHEN** user runs `edm analyze track.mp3` without `--profile` flag
- **THEN** profiling code paths SHALL not execute
- **AND** performance SHALL match baseline (no decorators or checks)

#### Scenario: Profiling decorator with profiling disabled
- **WHEN** function has `@profile_function` decorator and `EDM_PROFILE` is not set
- **THEN** decorator SHALL return function unchanged (zero overhead)
