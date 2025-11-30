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

## Open Questions
- Storage format for baselines? (JSON per commit, SQLite for trends, CSV?)
- Flamegraph generation? (require external tool like FlameGraph.pl or use library?)
- Should profiling be opt-in flag on existing commands or separate /profile command?
- How to handle non-deterministic timing in tests?
