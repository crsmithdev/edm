---
status: ready
created: 2025-12-05
updated: 2025-12-05
---

# [SECRACE] Fix Race Condition in Global Audio Cache

## Why

**Data Integrity Issue**: `src/edm/io/audio.py:123-142` uses a global audio cache without thread safety, creating race conditions in multi-worker mode.

**Current Behavior**:
- CLI uses `--workers N` for parallel analysis
- Multiple processes access shared global cache
- No locks or synchronization

**Risk**:
- Cache corruption
- Crashes from concurrent access
- Incorrect audio data returned

**Severity**: HIGH - Affects data integrity in production use case (parallel processing).

## What

### Files Affected
- `src/edm/io/audio.py` - Global `AudioCache` instance
- `src/edm/processing/parallel.py` - Multi-worker orchestration

### Specs Affected
- Audio processing specification (thread safety requirements)

## Impact

### Breaking Changes
None - API remains the same, only internal implementation changes.

### Migration Required
None.

### Risks
- **Threading overhead**: Minimal (locks only during cache access)
- **Alternative**: Use process-local caches (no shared state)
- **Testing**: Add concurrent access tests

### Dependencies
None - uses stdlib `threading.Lock`.
