# CLI Specification

## ADDED Requirements

### Requirement: Command-Line Entry Point
The CLI SHALL provide an executable entry point named `edm` that users can invoke from the terminal.

#### Scenario: Display help information
- **WHEN** user runs `edm --help`
- **THEN** displays usage information, available commands, and options

#### Scenario: Display version
- **WHEN** user runs `edm --version`
- **THEN** displays the version number of the CLI and library

#### Scenario: Run without arguments
- **WHEN** user runs `edm` with no arguments
- **THEN** displays brief usage message and available commands

### Requirement: Analyze Command
The CLI SHALL provide an `analyze` subcommand for analyzing audio tracks with a `--types` flag to specify which analyses to perform.

#### Scenario: Analyze single track with all analyses
- **WHEN** user runs `edm analyze track.mp3`
- **THEN** performs all available analyses (BPM and grid) and displays results in human-readable format

#### Scenario: Analyze with specific types
- **WHEN** user runs `edm analyze track.mp3 --types bpm`
- **THEN** performs only BPM analysis

#### Scenario: Analyze with multiple types
- **WHEN** user runs `edm analyze track.mp3 --types bpm,grid`
- **THEN** performs BPM and grid analyses

#### Scenario: Save results to file
- **WHEN** user runs `edm analyze track.mp3 --output results.json`
- **THEN** writes analysis results to the specified JSON file

#### Scenario: JSON output format
- **WHEN** user runs `edm analyze track.mp3 --format json`
- **THEN** displays results as JSON to stdout

#### Scenario: Invalid input file
- **WHEN** user runs `edm analyze nonexistent.mp3`
- **THEN** displays error message and exits with non-zero status code

#### Scenario: Invalid analysis type
- **WHEN** user runs `edm analyze track.mp3 --types invalid`
- **THEN** displays error listing valid analysis types and exits with non-zero status code

### Requirement: Progress Indication
The CLI SHALL display progress feedback for long-running operations.

#### Scenario: Show progress bar
- **WHEN** user runs analysis that takes more than 2 seconds
- **THEN** displays a progress bar showing analysis progress

#### Scenario: Verbose mode
- **WHEN** user runs `edm analyze track.mp3 --verbose`
- **THEN** sets logging level to DEBUG and shows detailed analysis steps in logs

#### Scenario: Quiet mode
- **WHEN** user runs `edm analyze track.mp3 --quiet`
- **THEN** suppresses all CLI output except final results and errors

### Requirement: Configuration File Support
The CLI SHALL support loading configuration from a file.

#### Scenario: Use default config location
- **WHEN** user runs `edm analyze track.mp3` and `~/.config/edm/config.toml` exists
- **THEN** loads configuration from the default location

#### Scenario: Specify custom config
- **WHEN** user runs `edm analyze track.mp3 --config my-config.toml`
- **THEN** loads configuration from the specified file

#### Scenario: Command-line options override config file
- **WHEN** user runs `edm analyze track.mp3 --config config.toml --types bpm`
- **THEN** command-line option takes precedence over config file setting

### Requirement: Error Handling and User Feedback
The CLI SHALL provide clear, actionable error messages to users.

#### Scenario: File not found error
- **WHEN** user specifies a non-existent audio file
- **THEN** displays error message indicating the file was not found and suggests checking the path

#### Scenario: Unsupported file format
- **WHEN** user specifies an unsupported audio format
- **THEN** displays error listing supported formats (MP3, WAV, FLAC, M4A)

#### Scenario: Missing dependencies
- **WHEN** required library dependency is not installed
- **THEN** displays error with installation instructions

#### Scenario: Configuration error
- **WHEN** configuration file contains invalid settings
- **THEN** displays error indicating which setting is invalid and expected format

### Requirement: Output Formatting with Rich Library
The CLI SHALL use the Rich library for formatted, readable terminal output.

#### Scenario: Tabular output
- **WHEN** user analyzes multiple tracks with `edm analyze *.mp3`
- **THEN** displays results in a Rich-formatted table

#### Scenario: Color-coded output
- **WHEN** terminal supports ANSI colors and `--no-color` flag is not set
- **THEN** uses Rich styling to highlight important information (errors in red, success in green)

#### Scenario: Disable colors for automation
- **WHEN** user runs `edm analyze track.mp3 --no-color`
- **THEN** disables all Rich formatting and outputs plain text

#### Scenario: Auto-detect TTY
- **WHEN** stdout is not a TTY (e.g., piped to another command)
- **THEN** automatically disables colors and progress bars

### Requirement: Batch Processing
The CLI SHALL support analyzing multiple files in a single invocation.

#### Scenario: Analyze directory
- **WHEN** user runs `edm analyze /path/to/tracks/ --recursive`
- **THEN** analyzes all audio files in the directory and subdirectories

#### Scenario: Glob pattern support
- **WHEN** user runs `edm analyze "tracks/*.mp3"`
- **THEN** analyzes all matching files

#### Scenario: Continue on errors
- **WHEN** user runs batch analysis and one file fails
- **THEN** logs the error and continues processing remaining files

#### Scenario: Batch results summary
- **WHEN** batch analysis completes
- **THEN** displays summary showing total files, successful analyses, and failures

### Requirement: Library Integration
The CLI SHALL delegate all business logic to the core library.

#### Scenario: CLI calls library functions
- **WHEN** user runs any CLI command
- **THEN** CLI imports and calls corresponding library functions without duplicating logic

#### Scenario: CLI handles I/O only
- **WHEN** implementing a new CLI command
- **THEN** all audio processing and analysis is performed by library, CLI only handles argument parsing and output formatting

### Requirement: Logging vs CLI Output Separation
The CLI SHALL maintain a clean separation between logging (for debugging) and CLI output (for user information).

#### Scenario: CLI output contains user-facing information
- **WHEN** analysis runs successfully
- **THEN** CLI output shows inputs, results, progress, and timing information

#### Scenario: Logs contain detailed debugging information
- **WHEN** analysis runs with logging enabled
- **THEN** log files contain detailed computation steps, intermediate values, and diagnostic information

#### Scenario: No print statements for debugging
- **WHEN** implementing CLI or library code
- **THEN** all debugging and detailed information uses the logging system, not print statements

#### Scenario: Log level controls detail
- **WHEN** user sets `--verbose` flag
- **THEN** logging level is set to DEBUG and detailed logs are written

### Requirement: Exit Codes
The CLI SHALL return appropriate exit codes for different outcomes.

#### Scenario: Successful execution
- **WHEN** analysis completes without errors
- **THEN** exits with code 0

#### Scenario: User error
- **WHEN** user provides invalid arguments or file not found
- **THEN** exits with code 1

#### Scenario: System error
- **WHEN** unhandled exception or system error occurs
- **THEN** exits with code 2

#### Scenario: Partial success in batch
- **WHEN** batch processing completes with some failures
- **THEN** exits with code 3

### Requirement: Performance Profiling and Timing
The CLI SHALL report timing information for all analysis operations.

#### Scenario: Single track timing
- **WHEN** user analyzes a single track
- **THEN** CLI displays total analysis time and breakdown by analysis type

#### Scenario: Batch timing summary
- **WHEN** user analyzes multiple tracks in batch
- **THEN** CLI displays total time, average time per track, and per-track timing details

#### Scenario: Timing included in JSON output
- **WHEN** user requests JSON format output
- **THEN** timing information is included in the JSON structure

#### Scenario: Profiling data in logs
- **WHEN** verbose logging is enabled
- **THEN** detailed profiling data for each analysis step is written to logs
