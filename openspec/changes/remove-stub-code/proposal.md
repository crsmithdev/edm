# Change: Remove Stub Code

## Why

The codebase contains stub implementations with placeholder TODO comments that create architectural confusion and technical debt:

1. **Features Module** (`src/edm/features/`) - spectral and temporal feature extraction functions return placeholder zeros, never used in analysis pipeline
2. **Models Module** (`src/edm/models/`) - abstract BaseModel class and load_model function that always raises ModelNotFoundError, never used

These stubs were scaffolded in the initial architecture but are not integrated into the actual analysis pipeline. The architecture review identified this as a critical issue reducing code clarity and maintainability.

**Why not Beatport/TuneBat?** These external service clients are also stubs, but the `replace-spotify-api` proposal will implement them. They should remain as placeholders until that proposal is applied.

**Why remove instead of implement?**
- **Features Module**: BPM analysis currently uses beat_this and librosa directly, not through a feature extraction abstraction. Adding this layer provides no current value and increases complexity.
- **Models Module**: No ML models are currently planned that need this abstraction. beat_this handles its own model loading internally.

Removing these stubs:
- Reduces cognitive overhead for developers
- Eliminates misleading API surface (documented but non-functional)
- Clarifies actual system capabilities
- Removes untested code paths
- Aligns code with specifications

## What Changes

1. **Remove Features Module**
   - Delete `src/edm/features/spectral.py`
   - Delete `src/edm/features/temporal.py`
   - Update `src/edm/features/__init__.py` to be empty or add note
   - Remove imports from any referencing code

2. **Remove Models Module**
   - Delete `src/edm/models/base.py`
   - Update `src/edm/models/__init__.py` to be empty or add note
   - Remove imports from any referencing code
   - Keep `ModelNotFoundError` in exceptions module for future use

3. **Update Documentation**
   - Remove references to feature extraction and model management from user-facing docs
   - Update architecture documentation to reflect removed modules
   - Note in changelog/migration guide (though these were never functional)

4. **Update Specifications**
   - Remove "Feature Extraction Module" requirement from core-library spec
   - Remove "Model Management Module" requirement from core-library spec
   - Keep External Data Retrieval requirement (Beatport/TuneBat will be implemented)

## Impact

- **Affected specs**: `core-library` (remove 2 requirements)
- **Affected code**:
  - Deleted: `src/edm/features/spectral.py`
  - Deleted: `src/edm/features/temporal.py`
  - Modified: `src/edm/features/__init__.py` (empty module or note)
  - Deleted: `src/edm/models/base.py`
  - Modified: `src/edm/models/__init__.py` (empty module or note)
  - Modified: `docs/architecture.md` (remove references)
- **Dependencies**: No dependency changes
- **User experience**: No impact - these functions were never functional
- **Backward compatibility**: Breaking API change, but affected APIs never worked
- **Tests**: No tests to update (no tests existed for stubs)

## Design Decisions

### Why Remove Entire Modules vs Mark as NotImplementedError?

**Considered Alternatives:**
1. Keep stubs, raise `NotImplementedError` on call
2. Keep stubs with TODO comments
3. Remove modules entirely

**Decision: Remove modules entirely**

**Rationale:**
- Stubs with `NotImplementedError` still appear in API documentation and IDE autocomplete
- Developers waste time discovering APIs don't work
- Empty modules with clear docstrings ("not yet implemented") are better than misleading functional-looking APIs
- Can always add back when needed with proper implementation

### Why Keep the Module Directories?

Keep `src/edm/features/` and `src/edm/models/` directories with minimal `__init__.py` to:
- Preserve import paths (prevents import errors if code tries to import from module)
- Document future plans in docstrings
- Make it clear these are intentionally empty, not accidentally missing

### Module __init__.py Content

```python
# src/edm/features/__init__.py
"""Feature extraction module (not yet implemented).

This module is reserved for future audio feature extraction implementations.
Current analysis uses librosa and beat_this directly.
"""

# src/edm/models/__init__.py
"""Model management module (not yet implemented).

This module is reserved for future ML model loading and management.
Current analysis uses beat_this with its own model loading.
"""
```

### Impact on replace-spotify-api Proposal

This change is complementary and independent:
- `replace-spotify-api` implements Beatport and TuneBat clients
- This change removes unused features and models modules
- No conflicts or dependencies between the two

### Future Re-addition Strategy

If feature extraction or model management is needed later:
1. Create new OpenSpec proposal documenting requirements
2. Implement with full test coverage
3. Integrate into analysis pipeline
4. Update specs with working scenarios

Don't add placeholder code speculatively.
