# Implementation Tasks

## 1. Setup uv Configuration

- [x] Add `.python-version` file specifying Python 3.9+ (or preferred version)
- [x] Ensure `pyproject.toml` is properly configured (already done)
- [x] Generate initial `uv.lock` file with `uv lock`
- [x] Add `uv.lock` to git
- [x] Add `.venv/` to `.gitignore` (if not already)

## 2. Update Documentation

- [x] Update `README.md` installation section to replace `pip` usage with `uv` installation and usage.
- [x] Update `README.md` installation to replace `pip` usage with `uv`
- [x] Replace `python -m venv .venv` with `uv venv` or note that uv manages it automatically
- [x] Replace `pip install` commands with `uv pip install` or `uv sync`
- [x] Update development installation instructions

## 3. Update CI/CD (If Exists)

- [x] Update GitHub Actions workflows to install and use uv
- [x] Replace pip caching with uv caching
- [x] Update test runners to use uv-managed environment

## 4. Testing and Validation

- [x] Test fresh installation on clean machine/container
- [x] Verify all dependencies install correctly
- [x] Ensure development dependencies work
- [x] Test that madmom installs from git correctly with uv
- [x] Verify CLI entry point works
- [x] Run existing test suite
