# Implementation Tasks

## Phase 1: Remove Features Module

### 1. Delete Feature Extraction Files
- [ ] Delete `src/edm/features/spectral.py`
- [ ] Delete `src/edm/features/temporal.py`
- [ ] Verify no imports exist in codebase (`rg "from edm.features import|from edm.features.spectral|from edm.features.temporal"`)

### 2. Update Features Module Init
- [ ] Update `src/edm/features/__init__.py` with minimal docstring
- [ ] Remove any exports of spectral/temporal functions
- [ ] Add note that module is reserved for future use

## Phase 2: Remove Models Module

### 3. Delete Model Management Files
- [ ] Delete `src/edm/models/base.py`
- [ ] Verify no imports of `BaseModel` exist (`rg "from edm.models.base import|BaseModel"`)
- [ ] Verify no imports of `load_model` exist (`rg "from edm.models import load_model|load_model\("`)

### 4. Update Models Module Init
- [ ] Update `src/edm/models/__init__.py` with minimal docstring
- [ ] Remove any exports of base model classes
- [ ] Add note that module is reserved for future use
- [ ] Keep `ModelNotFoundError` in exceptions module (already there)

### 5. Verify Exception Handling
- [ ] Check that `ModelNotFoundError` is defined in `src/edm/exceptions.py`
- [ ] If not, add it for future use
- [ ] Remove import of exception from deleted base.py if it exists

## Phase 3: Update Documentation

### 6. Update Architecture Documentation
- [ ] Remove Feature Extraction Module section from `docs/architecture.md`
- [ ] Remove Model Management Module section from `docs/architecture.md`
- [ ] Update module diagram/overview to reflect removed modules
- [ ] Add note that these may be added in future releases

### 7. Update API Documentation
- [ ] Verify no references to `extract_spectral_features` in docs
- [ ] Verify no references to `extract_temporal_features` in docs
- [ ] Verify no references to `load_model` in user-facing docs
- [ ] Update README.md if it mentions these features

### 8. Check Examples and Code Comments
- [ ] Search for example usage of features module (`rg "extract_spectral|extract_temporal"`)
- [ ] Search for example usage of models module (`rg "load_model"`)
- [ ] Remove or update any examples referencing these modules

## Phase 4: Validation and Testing

### 9. Verify Imports and Tests
- [ ] Run `rg "edm.features" --type py` to find any remaining imports
- [ ] Run `rg "edm.models" --type py` to find any remaining imports
- [ ] Verify no tests reference deleted modules
- [ ] Run full test suite: `uv run pytest`
- [ ] Verify all tests pass

### 10. Type Checking and Linting
- [ ] Run mypy: `uv run mypy src/`
- [ ] Verify no type errors from missing imports
- [ ] Run ruff: `uv run ruff check .`
- [ ] Fix any linting issues

### 11. Import Validation
- [ ] Try importing edm package: `python -c "import edm"`
- [ ] Verify no import errors
- [ ] Try importing submodules: `python -c "import edm.features; import edm.models"`
- [ ] Verify modules exist but are empty

## Phase 5: Final Verification

### 12. Manual Testing
- [ ] Run CLI with a sample file: `uv run edm analyze tests/fixtures/test.mp3 --no-color`
- [ ] Verify analysis works without errors
- [ ] Check that no warnings about missing features/models appear

### 13. Documentation Review
- [ ] Verify `docs/architecture.md` reflects current state
- [ ] Verify no broken links to feature extraction or model management
- [ ] Update project structure diagram if needed

### 14. Git Housekeeping
- [ ] Review all changed files
- [ ] Ensure no unintended deletions
- [ ] Verify empty __init__.py files have appropriate content

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
