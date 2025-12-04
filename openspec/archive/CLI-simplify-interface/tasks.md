# Tasks: CLI Simplification

## 1. Analyze Command Cleanup

- [x] 1.1 Remove `--json-logs` flag from main.py
- [x] 1.2 Remove `--structure-detector` flag from main.py
- [x] 1.3 Rename `--ignore-metadata` to `--no-metadata`
- [x] 1.4 Replace `--log-level` with `-v`/`-vv` verbosity counting
- [x] 1.5 Fix `--workers` to cap at min(max_workers, file_count)
- [x] 1.6 Keep `--offline`, `--annotations`, `--output` (used by workflow)

## 2. Evaluate Command Cleanup

- [x] 2.1 Remove `--tolerance` flag (hardcode default 2.0s)
- [x] 2.2 Remove `--detector` flag (hardcode auto/msaf)
- [x] 2.3 Keep `--output`, `--reference` (used by workflow)

## 3. Profile Command Removal

- [x] 3.1 Delete `src/cli/commands/profile.py`
- [x] 3.2 Delete `src/edm/profiling/` directory
- [x] 3.3 Remove profile command registration from `src/cli/main.py`

## 4. Implementation Details

- [x] 4.1 Map verbosity count to log levels (0=WARNING, 1=INFO, 2=DEBUG)
- [x] 4.2 Update help text for modified flags
- [x] 4.3 Remove json_logs parameter, hardcode to False

## 5. Testing

- [x] 5.1 Test removed flags error appropriately (`--json-logs` errors)
- [x] 5.2 Verify `-v` flag shows in help
- [x] 5.3 Verify `--no-metadata` in help
- [x] 5.4 Verify `--tolerance` and `--detector` removed from evaluate
- [ ] 5.5 Run full CLI tests: `pytest tests/cli/` (deferred)

## 6. Spec Updates

- [x] 6.1 Update `openspec/specs/cli/spec.md` with changes
- [x] 6.2 Document new verbosity convention
- [x] 6.3 Add notes about kept flags for workflow compatibility
