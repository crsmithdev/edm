## 1. Remove Feature Stubs

- [ ] 1.1 Delete `src/edm/features/spectral.py`
- [ ] 1.2 Delete `src/edm/features/temporal.py`
- [ ] 1.3 Delete `src/edm/features/__init__.py`
- [ ] 1.4 Remove `src/edm/features/` directory

## 2. Remove Model Stubs

- [ ] 2.1 Delete `src/edm/models/base.py`
- [ ] 2.2 Delete `src/edm/models/__init__.py`
- [ ] 2.3 Remove `src/edm/models/` directory

## 3. Remove External Service Stubs

- [ ] 3.1 Delete `src/edm/external/beatport.py`
- [ ] 3.2 Delete `src/edm/external/tunebat.py`
- [ ] 3.3 Update `src/edm/external/__init__.py` to remove exports

## 4. Update Structure Analysis

- [ ] 4.1 Add `implemented: bool = False` field to `StructureResult` dataclass
- [ ] 4.2 Modify `analyze_structure()` to return empty result with `implemented=False`
- [ ] 4.3 Update CLI to check `implemented` flag and display "not implemented" message
- [ ] 4.4 Keep structure columns in output but show "N/A" or similar

## 5. Cleanup

- [ ] 5.1 Update `src/edm/__init__.py` to remove feature/model imports if present
- [ ] 5.2 Remove any tests for deleted stub code
- [ ] 5.3 Run full test suite to verify no breakage
- [ ] 5.4 Run `ruff check` and `mypy` to catch dead imports
