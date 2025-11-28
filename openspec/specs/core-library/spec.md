# core-library Specification

## Purpose
TBD - created by archiving change add-core-architecture. Update Purpose after archive.
## Requirements
### Requirement: Package Structure
The core library SHALL be organized as a Python package named `edm` with modular subpackages for different functional areas.

#### Scenario: Import core analysis module
- **WHEN** a user imports `from edm.analysis import analyze_track`
- **THEN** the function is available without errors

#### Scenario: Import feature extraction module
- **WHEN** a user imports `from edm.features import extract_spectral_features`
- **THEN** the function is available without errors

#### Scenario: Package initialization
- **WHEN** a user imports `import edm`
- **THEN** the package imports successfully and exposes version information

### Requirement: Audio Analysis Module

The library SHALL provide an analysis module (`edm.analysis`) that contains functions for analyzing EDM tracks using beat_this for neural network-based BPM detection and librosa as fallback.

#### Scenario: Analyze track for BPM
- **WHEN** user calls `analyze_track(filepath, config)` with BPM detection enabled
- **THEN** returns a result object containing BPM value and confidence score

#### Scenario: Analyze track structure
- **WHEN** user calls `analyze_track(filepath, config)` with structure detection enabled
- **THEN** returns a result object containing detected sections (intro, drop, breakdown, etc.)

#### Scenario: Handle invalid audio file
- **WHEN** user calls `analyze_track(filepath, config)` with an invalid file path
- **THEN** raises a custom `AudioFileError` exception with descriptive message

#### Scenario: BPM computation method selection
- **WHEN** user calls `compute_bpm(filepath, prefer_madmom=True)`
- **THEN** uses beat_this (neural network approach) as primary detector with librosa fallback

#### Scenario: BPM computation with librosa preference
- **WHEN** user calls `compute_bpm(filepath, prefer_madmom=False)`
- **THEN** uses librosa (traditional DSP approach) as primary detector

### Requirement: File I/O Module
The library SHALL provide an I/O module (`edm.io`) for reading and writing audio files and metadata.

#### Scenario: Load audio file
- **WHEN** user calls `load_audio(filepath)` with a valid audio file
- **THEN** returns audio data and sample rate

#### Scenario: Support multiple audio formats
- **WHEN** user calls `load_audio(filepath)` with MP3, WAV, FLAC, or M4A files
- **THEN** successfully loads the audio data

#### Scenario: Read metadata from file
- **WHEN** user calls `read_metadata(filepath)` with an audio file
- **THEN** returns metadata dictionary with artist, title, album, and technical info

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports BPM lookup strategy configuration with file-based and programmatic options.

#### Scenario: Configure BPM lookup order
- **WHEN** user sets `config.bpm_lookup_strategy = ["metadata", "computed"]`
- **THEN** BPM analysis follows specified order

#### Scenario: Skip metadata lookup
- **WHEN** user sets `config.bpm_lookup_strategy = ["computed"]`
- **THEN** system skips metadata and computes BPM directly from audio

#### Scenario: Force computation via configuration
- **WHEN** user sets `config.bpm_force_compute = True`
- **THEN** all BPM lookups skip metadata, computing directly from audio

### Requirement: Type Safety
The library SHALL use Python type hints throughout the public API.

#### Scenario: Type checking with mypy
- **WHEN** mypy is run on library code
- **THEN** no type errors are reported

#### Scenario: IDE autocomplete support
- **WHEN** user types `analyze_track(` in an IDE
- **THEN** IDE shows parameter types and return type

### Requirement: Error Handling
The library SHALL define custom exception classes for different error categories.

#### Scenario: Audio file errors
- **WHEN** audio file cannot be loaded
- **THEN** raises `AudioFileError` with specific reason (not found, unsupported format, corrupted)

#### Scenario: Analysis errors
- **WHEN** analysis fails due to invalid input
- **THEN** raises `AnalysisError` with details about what failed

### Requirement: Logging Support
The library SHALL provide structured logging for debugging and monitoring.

#### Scenario: Enable debug logging
- **WHEN** user sets logging level to DEBUG
- **THEN** detailed analysis steps are logged

#### Scenario: Default logging behavior
- **WHEN** user does not configure logging
- **THEN** only warnings and errors are logged to stderr

### Requirement: Documentation
The library SHALL include docstrings for all public functions, classes, and modules using NumPy style.

#### Scenario: View function documentation
- **WHEN** user calls `help(analyze_track)` in Python REPL
- **THEN** displays formatted docstring with parameters, returns, and examples

#### Scenario: Generate API documentation
- **WHEN** Sphinx documentation is generated from docstrings
- **THEN** produces complete API reference documentation

### Requirement: Performance Profiling
The library SHALL provide built-in profiling capabilities to measure and report execution time for analysis operations.

#### Scenario: Profile analysis operations
- **WHEN** user calls `analyze_track()` with profiling enabled
- **THEN** returns timing information for each analysis step along with results

#### Scenario: Nested profiling context
- **WHEN** profiling is enabled and functions call other profiled functions
- **THEN** timing data is collected hierarchically showing parent-child relationships

#### Scenario: Access profiling data programmatically
- **WHEN** user calls analysis function
- **THEN** result object includes `timing` attribute with breakdown by operation

#### Scenario: Minimal overhead when profiling disabled
- **WHEN** profiling is disabled (default)
- **THEN** profiling code adds negligible performance overhead (< 1%)

### Requirement: Batch Processing Optimization
The library SHALL optimize BPM analysis for batch processing with caching and parallel support.

#### Scenario: Cache prevents redundant API calls
- **WHEN** analyzing multiple files from same artist/album
- **THEN** Spotify API is called once per unique track, using cache for duplicates

#### Scenario: Parallel metadata reading
- **WHEN** analyzing batch of 100+ files
- **THEN** metadata reading happens in parallel for improved performance

#### Scenario: Progress callback for batch operations
- **WHEN** analyzing batch with progress callback provided
- **THEN** callback is invoked with progress updates after each track

### Requirement: Shared Accuracy Utilities
The project SHALL provide shared utility functions in `scripts/accuracy/common.py` for accuracy evaluation tasks.

#### Scenario: Discover audio files
- **WHEN** script calls `discover_audio_files(source_path)`
- **THEN** returns list of all audio files (.mp3, .flac, .wav, .m4a) found recursively

#### Scenario: Handle missing ground truth gracefully
- **WHEN** script calls `sample_random(files, size, seed)`
- **THEN** returns reproducible random sample of specified size using provided seed
- **WHEN** ground truth is unavailable for a file
- **THEN** system logs warning, tracks count of missing ground truth, and excludes from metrics calculation
- **WHEN** script calls `load_ground_truth_csv(path, value_field)`
#### Scenario: Calculate accuracy metrics
The existing analysis module SHALL be used by accuracy evaluators for computation without modification.

#### Scenario: Reuse BPM analysis in evaluator
The existing analysis module is used by evaluation scripts for computation without modification.
- **THEN** calls `analyze_bpm(filepath, force_compute=True, ignore_metadata=True, offline=True)` to ensure pure computation without lookups

- **WHEN** BPM evaluation script evaluates a file
#### Scenario: Ensure reproducible sampling

