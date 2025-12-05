---
status: deployed
created: 2025-12-05
---

# [ARCHDEP] Break Circular Dependencies in Analysis Layer

## Why

`src/edm/analysis/structure.py` imports and calls `analyze_bpm()` and `detect_beats()`, creating circular dependencies.

**Impact**: Cannot test structure detection independently, fragile dependency chains, violates single responsibility.

## What

- `src/edm/analysis/structure.py` - Remove direct BPM/beat calls
- Create `src/edm/analysis/orchestrator.py` - Coordinate multi-stage analysis
- `src/cli/commands/analyze.py` - Use orchestrator

## Impact

Breaking: Internal API change (public CLI API unchanged).
