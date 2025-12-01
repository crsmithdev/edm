# Design: [PROF] Profiling Infrastructure

## Design Ideas & Approach
Layered profiling system:
1. **Lightweight layer**: Quick profiling for development (cProfile, simple timing)
2. **Detailed layer**: Deep analysis for optimization (py-spy, memory_profiler, tracemalloc)
3. **Comparison layer**: Baseline management and regression detection

Use a profiler abstraction that wraps multiple backends, allowing swappable implementations.

## Architecture & Components
- `src/edm/profiling/` - Core profiling module
  - `profiler.py` - Abstract profiler interface
  - `cpu.py` - CPU profiling (cProfile, py-spy)
  - `memory.py` - Memory profiling (tracemalloc, memory_profiler)
  - `io.py` - I/O analysis
  - `baseline.py` - Baseline storage and comparison
  - `reporter.py` - Output formatting (console, JSON, HTML)
- `src/cli/commands/profile.py` - CLI integration
- `tests/unit/test_profiling/` - Profiler tests
- `benchmarks/` - Baseline storage (JSON files per version/commit)
- `docs/profiling.md` - Usage guide

## Trade-offs & Decisions
**Why abstraction over direct tool usage:**
- Tools have different APIs and output formats
- Allows switching tools without code changes
- Makes testing easier with mock profilers

**Why multiple tools:**
- cProfile: Fast CPU profiling, part of stdlib
- py-spy: Low overhead, flamegraph output, production-safe
- tracemalloc: Memory tracking, line-level granularity
- Layers allow "quick check" (fast) vs "deep dive" (slower)

**Why baseline comparison:**
- Catch regressions early
- Document when/why performance changed
- Build historical performance trends

## Requirements
1. Profilers must support context managers (easy integration into code)
2. Results must be serializable (JSON for storage/comparison)
3. CLI should support profiling individual commands (analyze, evaluate)
4. Baseline format must be stable across versions
5. Low overhead when profiling is disabled

## Decisions

### Storage Format
**Decision**: JSON per commit/baseline
**Rationale**:
- Git-friendly (text format, diff-able)
- Human readable for debugging
- Simple to implement (stdlib json module)
- No external database dependencies
- Easy to version control alongside code

**Format**:
```json
{
  "commit": "abc123",
  "timestamp": "2025-12-01T12:00:00Z",
  "profiles": {
    "analyze_bpm": {
      "cpu_time": 2.5,
      "wall_time": 2.7,
      "peak_memory_mb": 450,
      "calls": 1
    }
  }
}
```

### Flamegraph Generation
**Decision**: Use py-spy's built-in flamegraph support
**Rationale**:
- py-spy natively generates flame graphs (no external tools)
- SVG output works in browsers, CI artifacts, docs
- Low overhead sampling profiler (safe for production)
- No dependency on FlameGraph.pl or d3-flamegraph

### CLI Integration
**Decision**: Decorator-based opt-in with `--profile` flag
**Rationale**:
- Minimal code changes to existing commands
- Keeps profiling orthogonal to business logic
- Users explicitly opt-in (no performance overhead by default)
- Allows profiling any command: `edm analyze track.mp3 --profile cpu`

**Example**:
```bash
edm analyze track.mp3 --profile cpu        # CPU profiling
edm analyze track.mp3 --profile memory     # Memory profiling
edm analyze track.mp3 --profile cpu,memory # Both
```

### Non-Deterministic Timing in Tests
**Decision**: Relative regression thresholds with tolerance
**Rationale**:
- Absolute times vary by machine (CI vs laptop vs desktop)
- Focus on regressions (>20% slower than baseline)
- Use median of 3 runs to reduce noise
- Mark performance tests with `@pytest.mark.slow` for optional execution

**Test Strategy**:
- Unit tests: Mock profiler calls, test infrastructure only
- Integration tests: Verify profiler captures data, don't assert timing
- Performance tests: Compare against baseline with 20% tolerance
- Benchmark suite: Store baselines in `benchmarks/` directory
