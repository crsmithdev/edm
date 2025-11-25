# Implementation Tasks

## 1. Project Structure Setup
- [x] 1.1 Create `src/edm/` directory for core library
- [x] 1.2 Create `src/cli/` directory for CLI application
- [x] 1.3 Add `__init__.py` files for Python packages
- [x] 1.4 Create `pyproject.toml` with dependencies (typer, rich, spotipy, requests, librosa, madmom, essentia)
- [x] 1.5 Create `requirements.txt` for pip installation

## 2. Core Library Foundation
- [x] 2.1 Create `src/edm/analysis/` module with BPM and grid analysis stubs
- [x] 2.2 Create `src/edm/io/` module for audio file loading and metadata reading
- [x] 2.3 Create `src/edm/external/` module with Spotify, Beatport, and TuneBat clients
- [x] 2.4 Create `src/edm/models/` module for ML model loading
- [x] 2.5 Create `src/edm/features/` module for audio feature extraction
- [x] 2.6 Add base classes, custom exceptions, and return types (dataclasses)
- [x] 2.7 Implement profiling utilities for timing analysis operations

## 3. CLI Implementation with Rich and Typer
- [x] 3.1 Create `src/cli/main.py` as entry point using Typer
- [x] 3.2 Implement `analyze` command with `--types` flag for selecting analyses
- [x] 3.3 Implement Rich-based output formatting for tables and progress bars
- [x] 3.4 Add `--no-color` flag to disable Rich formatting
- [x] 3.5 Add `--verbose` flag for DEBUG logging level
- [x] 3.6 Add `--quiet` flag to suppress non-essential output
- [x] 3.7 Integrate CLI with core library API (delegate all logic to library)
- [x] 3.8 Implement timing display for single track and batch analysis

## 4. Logging System
- [x] 4.1 Set up Python logging module with appropriate formatters
- [x] 4.2 Configure log file location (~/.local/share/edm/logs/)
- [x] 4.3 Implement log level control via CLI flags
- [x] 4.4 Ensure all library code uses logging (never print statements)
- [x] 4.5 Document separation: CLI output (Rich) vs logs (debugging)

## 5. Configuration and Setup
- [x] 5.1 Create configuration system using Pydantic models
- [x] 5.2 Support TOML configuration files
- [x] 5.3 Implement config file loading from default and custom locations
- [x] 5.4 Add command-line override support for config values

## 6. External API Integration
- [x] 6.1 Implement Spotify API client using spotipy
- [x] 6.2 Implement Beatport API client (or web scraper if no public API)
- [x] 6.3 Implement TuneBat API client (or web scraper)
- [x] 6.4 Add request caching to avoid redundant API calls
- [x] 6.5 Implement error handling for API timeouts and rate limits

## 7. Testing Infrastructure
- [x] 7.1 Create `tests/` directory structure (unit/ and integration/)
- [x] 7.2 Add pytest configuration
- [x] 7.3 Create test fixtures for sample audio files
- [x] 7.4 Write unit tests for library modules (analysis, io, external)
- [x] 7.5 Write integration tests for CLI commands
- [x] 7.6 Test profiling accuracy and overhead

## 8. Documentation
- [x] 8.1 Add NumPy-style docstrings to all public functions
- [x] 8.2 Create README.md with installation and usage examples
- [x] 8.3 Document CLI `analyze` command with all flags
- [x] 8.4 Document logging vs CLI output separation
- [x] 8.5 Document external API setup (credentials, rate limits)
