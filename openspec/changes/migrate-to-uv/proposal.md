# Change: Migrate from pip to uv for dependency management

## Why

The current development workflow uses pip and manual virtualenv management, which is slow and requires multiple manual steps. uv is a modern, extremely fast Python package installer and resolver written in Rust that can replace pip and manage virtual environments automatically, reducing installation times from minutes to seconds and simplifying the developer experience.

## What Changes

- Replace pip commands with uv in all documentation and workflows
- Add uv lockfile (`uv.lock`) for reproducible builds
- Let uv automatically manage virtual environments instead of manual `python -m venv` creation
- Update CI/CD workflows to use uv
- Maintain backward compatibility with existing `pyproject.toml`
- Add `.python-version` file for consistent Python version management

## Impact

- **Affected specs**: build-tooling (new capability)
- **Affected code**: 
  - `README.md` (installation instructions)
  - `.github/workflows/` (if CI exists)
  - Development documentation
- **Developer experience**: Faster installs (10-100x speedup), simpler workflow, automatic virtualenv management
- **No breaking changes**: Project structure and pyproject.toml remain compatible
