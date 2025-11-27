# Implementation Tasks

## Phase 1: Remove Features Module

### 1. Delete Feature Extraction Files
- [x] Delete `src/edm/features/spectral.py`
- [x] Delete `src/edm/features/temporal.py`
- [x] Verify no imports exist in codebase (`rg "from edm.features import|from edm.features.spectral|from edm.features.temporal"`)

### 2. Update Features Module Init
- [x] Update `src/edm/features/__init__.py` with minimal docstring
- [x] Remove any exports of spectral/temporal functions
- [x] Add note that module is reserved for future use

## Phase 2: Remove Models Module

### 3. Delete Model Management Files
- [x] Delete `src/edm/models/base.py`
- [x] Verify no imports of `BaseModel` exist (`rg "from edm.models.base import|BaseModel"`)
- [x] Verify no imports of `load_model` exist (`rg "from edm.models import load_model|load_model\("`)

### 4. Update Models Module Init
- [x] Update `src/edm/models/__init__.py` with minimal docstring
- [x] Remove any exports of base model classes
- [x] Add note that module is reserved for future use
- [x] Keep `ModelNotFoundError` in exceptions module (already there)

### 5. Verify Exception Handling
- [x] Check that `ModelNotFoundError` is defined in `src/edm/exceptions.py`
- [x] If not, add it for future use
- [x] Remove import of exception from deleted base.py if it exists

## Phase 3: Update Documentation

### 6. Update Architecture Documentation
- [x] Remove Feature Extraction Module section from `docs/architecture.md`
- [x] Remove Model Management Module section from `docs/architecture.md`
- [x] Update module diagram/overview to reflect removed modules
- [x] Add note that these may be added in future releases

### 7. Update API Documentation
- [x] Verify no references to `extract_spectral_features` in docs
- [x] Verify no references to `extract_temporal_features` in docs
- [x] Verify no references to `load_model` in user-facing docs
- [x] Update README.md if it mentions these features

### 8. Check Examples and Code Comments
- [x] Search for example usage of features module (`rg "extract_spectral|extract_temporal"`)
- [x] Search for example usage of models module (`rg "load_model"`)
- [x] Remove or update any examples referencing these modules

## Phase 4: Validation and Testing

### 9. Verify Imports and Tests
- [x] Run `rg "edm.features" --type py` to find any remaining imports
- [x] Run `rg "edm.models" --type py` to find any remaining imports
- [x] Verify no tests reference deleted modules
- [x] Run full test suite: `uv run pytest`
- [x] Verify all tests pass

### 10. Type Checking and Linting
- [x] Run mypy: `uv run mypy src/`
- [x] Verify no type errors from missing imports
- [x] Run ruff: `uv run ruff check .`
- [x] Fix any linting issues

### 11. Import Validation
- [x] Try importing edm package: `python -c "import edm"`
- [x] Verify no import errors
- [x] Try importing submodules: `python -c "import edm.features; import edm.models"`
- [x] Verify modules exist but are empty

## Phase 5: Final Verification

### 12. Manual Testing
- [x] Run CLI with a sample file: `uv run edm analyze tests/fixtures/test.mp3 --no-color`
- [x] Verify analysis works without errors
- [x] Check that no warnings about missing features/models appear

### 13. Documentation Review
- [x] Verify `docs/architecture.md` reflects current state
- [x] Verify no broken links to feature extraction or model management
- [x] Update project structure diagram if needed

### 14. Git Housekeeping
- [x] Review all changed files
- [x] Ensure no unintended deletions
- [x] Verify empty __init__.py files have appropriate content

## Dependencies and Parallelization

**Sequential Tasks:**
- Tasks 1-2 must complete before Phase 2 begins
- Tasks 3-5 must complete before Phase 3 begins
- Phase 4 tasks depend on all code changes being complete

**Parallelizable:**
- Task 6 (architecture docs) can run in parallel with Task 7 (API docs)
- Tasks 9-11 (validation) can be run concurrently

**Blocking Tasks:**
- Task 5 (verify exception handling) must complete before tests run
- Task 12 (manual testing) depends on all code changes
