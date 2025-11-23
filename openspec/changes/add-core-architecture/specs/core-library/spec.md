# Core Library Specification

## ADDED Requirements

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
The library SHALL provide an analysis module (`edm.analysis`) that contains functions for analyzing EDM tracks.

#### Scenario: Analyze track for BPM
- **WHEN** user calls `analyze_track(filepath, config)` with BPM detection enabled
- **THEN** returns a result object containing BPM value and confidence score

#### Scenario: Analyze track structure
- **WHEN** user calls `analyze_track(filepath, config)` with structure detection enabled
- **THEN** returns a result object containing detected sections (intro, drop, breakdown, etc.)

#### Scenario: Handle invalid audio file
- **WHEN** user calls `analyze_track(filepath, config)` with an invalid file path
- **THEN** raises a custom `AudioFileError` exception with descriptive message

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

### Requirement: External Data Retrieval Module
The library SHALL provide an external module (`edm.external`) for retrieving BPM and track information from Spotify, Beatport, and TuneBat.

#### Scenario: Query Spotify for track info
- **WHEN** user calls `search_spotify(artist, title)` with valid parameters
- **THEN** returns matching track information including BPM if available from Spotify API

#### Scenario: Query Beatport for BPM
- **WHEN** user calls `search_beatport(artist, title)` with valid parameters
- **THEN** returns BPM and key information from Beatport

#### Scenario: Query TuneBat for BPM
- **WHEN** user calls `search_tunebat(artist, title)` with valid parameters
- **THEN** returns BPM, key, and other analysis data from TuneBat

#### Scenario: Handle API timeout
- **WHEN** external API request exceeds timeout threshold
- **THEN** raises a custom `ExternalServiceError` with retry information

#### Scenario: Cache external requests
- **WHEN** the same external data request is made within cache lifetime
- **THEN** returns cached result without making new API call

#### Scenario: Aggregate results from multiple sources
- **WHEN** user calls `get_track_info(artist, title)` without specifying source
- **THEN** queries all available sources and returns aggregated results with source attribution

### Requirement: Feature Extraction Module
The library SHALL provide a features module (`edm.features`) for extracting audio features used in analysis.

#### Scenario: Extract spectral features
- **WHEN** user calls `extract_spectral_features(audio_data, sample_rate)`
- **THEN** returns features including spectral centroid, rolloff, and flux

#### Scenario: Extract temporal features
- **WHEN** user calls `extract_temporal_features(audio_data, sample_rate)`
- **THEN** returns features including RMS energy, zero-crossing rate, and onset strength

### Requirement: Model Management Module
The library SHALL provide a models module (`edm.models`) for loading and managing ML models.

#### Scenario: Load pre-trained model
- **WHEN** user calls `load_model(model_name)` with a valid model identifier
- **THEN** loads the model and returns a model instance ready for inference

#### Scenario: Model not found
- **WHEN** user calls `load_model(model_name)` with an invalid identifier
- **THEN** raises a custom `ModelNotFoundError` exception

### Requirement: Configuration Management
The library SHALL provide a configuration system that supports both file-based and programmatic configuration.

#### Scenario: Load config from file
- **WHEN** user calls `load_config(filepath)` with a valid TOML config file
- **THEN** returns a validated configuration object

#### Scenario: Use default configuration
- **WHEN** user creates `AnalysisConfig()` without parameters
- **THEN** returns configuration object with sensible defaults

#### Scenario: Override with environment variables
- **WHEN** environment variable `EDM_MODEL_PATH` is set
- **THEN** configuration uses environment value instead of default

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

#### Scenario: External service errors
- **WHEN** external API call fails
- **THEN** raises `ExternalServiceError` with error details and retry information

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
