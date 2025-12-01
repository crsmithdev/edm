# [PROF] Add Profiling Infrastructure

## Overview
Establish production-ready profiling capabilities for performance analysis and optimization. Support multiple profiling types (CPU, memory, I/O) at different levels (function, module, full application), with tooling to compare baseline vs current performance.

## User Value
- Identify performance bottlenecks systematically before they become production issues
- Track performance regressions across changes
- Optimize hot paths with data-driven decisions
- Build institutional knowledge of performance characteristics as the codebase grows

## Scope
- CPU profiling (cProfile, py-spy)
- Memory profiling (tracemalloc, memory_profiler)
- I/O profiling (call stack analysis)
- Baseline collection and comparison tooling
- Integration with pytest for benchmark tests
- CLI commands for profiling analysis tasks

Out of scope:
- GPU profiling (defer until torch usage expands)
- Distributed tracing
- Real-time monitoring dashboards

## Status
- [x] Design approved
- [x] Full proposal created (specs + tasks)
- [ ] Implementation complete

## Implementation Summary

**Core Capabilities:**
1. CPU profiling with cProfile and py-spy flamegraphs
2. Memory profiling with tracemalloc
3. Baseline storage and regression detection (JSON format)
4. CLI integration via `--profile` flag on analyze/evaluate commands
5. Decorator-based function profiling with zero overhead when disabled
6. pytest benchmark integration

**Key Design Decisions:**
- JSON-based baseline storage (git-friendly, human-readable)
- py-spy for flamegraph generation (built-in, no external tools)
- Opt-in via CLI flags and decorators (no performance impact by default)
- Relative regression thresholds (20% tolerance) for non-deterministic timing

See `design.md` for architecture details and `tasks.md` for implementation steps.
