## MODIFIED Requirements

### Requirement: Batch Processing
The CLI SHALL support analyzing multiple files in parallel using multiprocessing by default.

#### Scenario: Analyze directory with default parallelism
- **WHEN** user runs `edm analyze /path/to/tracks/ --recursive`
- **THEN** analyzes all audio files using (CPU count - 1) parallel workers

#### Scenario: Analyze multiple files with automatic parallelism
- **WHEN** user runs `edm analyze *.mp3` with 10 files on an 8-core system
- **THEN** uses 7 parallel workers to process files concurrently

#### Scenario: Override to single-threaded execution
- **WHEN** user runs `edm analyze *.mp3 --workers 1`
- **THEN** processes files one at a time using single worker

#### Scenario: Glob pattern support
- **WHEN** user runs `edm analyze "tracks/*.mp3"`
- **THEN** analyzes all matching files in parallel

#### Scenario: Continue on errors
- **WHEN** user runs batch analysis and one file fails
- **THEN** logs the error and continues processing remaining files

#### Scenario: Batch results summary
- **WHEN** batch analysis completes
- **THEN** displays summary showing total files, successful analyses, and failures

### Requirement: Progress Indication
The CLI SHALL display progress feedback with completion counts for long-running operations.

#### Scenario: Show progress bar with worker count
- **WHEN** user runs analysis with multiple files
- **THEN** displays progress bar with spinner, bar, completion ratio (e.g., "12/50"), and elapsed time

#### Scenario: Show progress bar for single worker
- **WHEN** user runs `edm analyze *.mp3 --workers 1`
- **THEN** displays progress bar without "N workers" notation

#### Scenario: Verbose mode
- **WHEN** user runs `edm analyze track.mp3 --verbose`
- **THEN** sets logging level to DEBUG and shows detailed analysis steps in logs

#### Scenario: Quiet mode
- **WHEN** user runs `edm analyze track.mp3 --quiet`
- **THEN** suppresses progress bar and non-essential output

## REMOVED Requirements

### Requirement: Sequential Processing Mode
**Reason**: Redundant with parallel processing using workers=1. Maintaining two code paths added unnecessary complexity.

**Migration**: Users who previously relied on default sequential behavior will now get parallel processing by default (better performance). Users who need single-threaded execution can use `--workers 1`.
