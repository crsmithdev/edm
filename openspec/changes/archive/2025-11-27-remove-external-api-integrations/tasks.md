## 1. Remove External Module

- [x] 1.1 Delete `src/edm/external/spotify.py`
- [x] 1.2 Delete `src/edm/external/getsongbpm.py`
- [x] 1.3 Delete `src/edm/external/__init__.py`
- [x] 1.4 Delete `src/edm/external/` directory
- [x] 1.5 Delete `tests/unit/test_getsongbpm.py`

## 2. Update BPM Analysis

- [x] 2.1 Remove external lookup from `src/edm/analysis/bpm.py`
- [x] 2.2 Simplify `BPMStrategy` to only support metadata and computed
- [x] 2.3 Update strategy docstrings

## 3. Update Configuration

- [x] 3.1 Remove `ExternalServicesConfig` class from `src/edm/config.py`
- [x] 3.2 Remove `external_services` field from `EDMConfig`
- [x] 3.3 Update `bpm_lookup_strategy` default to `["metadata", "computed"]`
- [x] 3.4 Remove `dotenv` import and `load_dotenv()` call
- [x] 3.5 Update config docstrings

## 4. Remove Dependencies

- [x] 4.1 Remove `spotipy` from `pyproject.toml`
- [x] 4.2 Remove `python-dotenv` from `pyproject.toml`
- [x] 4.3 Run `uv sync` to update lockfile

## 5. Cleanup Obsolete Proposal

- [x] 5.1 Delete `openspec/changes/replace-spotify-api/` directory

## 6. Update Documentation

- [x] 6.1 Update `README.md` to remove API setup instructions
- [x] 6.2 Update `docs/architecture.md` to reflect simplified design
- [x] 6.3 Update `docs/cli-reference.md` if needed

## 7. Testing

- [x] 7.1 Run full test suite
- [x] 7.2 Verify BPM analysis works with metadata and computed only
- [x] 7.3 Run mypy type checking
- [x] 7.4 Run ruff linting
