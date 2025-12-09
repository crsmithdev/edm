# Integration Tests

End-to-end tests that verify multiple components working together.

## Training Smoke Test

**File:** `test_training_smoke.py`
**Purpose:** Catch training pipeline bugs before they reach production
**Runtime:** ~10-15 seconds (GPU), ~20-30 seconds (CPU)

### What it tests:

- Dataset loading and label vocabulary auto-detection
- Model creation with dynamic num_classes
- Forward/backward pass without crashes
- Loss computation for all heads
- Checkpoint saving/loading
- Model inference after loading

### Running the tests:

```bash
# Run all slow/integration tests
uv run pytest tests/integration/test_training_smoke.py -v -m slow

# Run specific test
uv run pytest tests/integration/test_training_smoke.py::test_training_smoke -v -m slow

# Run without coverage (faster)
uv run pytest tests/integration/test_training_smoke.py -v -m slow --no-cov
```

### CI/Pre-commit usage:

**Recommended:** Run before commits that touch training code:

```bash
# In .git/hooks/pre-commit or CI
uv run pytest tests/integration/test_training_smoke.py -m slow --no-cov -q
```

The `-q` flag makes output minimal. Exit code 0 = pass, non-zero = fail.

### What to do when it fails:

1. **Label vocabulary error** - New label in data, model num_classes mismatch fixed
2. **Shape mismatch** - Model architecture change broke compatibility
3. **Checkpoint load error** - Model structure changed, update loading logic
4. **Training crash** - Loss computation or optimizer issue

These tests use synthetic data (2 tracks, 5 seconds silence) so failures indicate code bugs, not data issues.
