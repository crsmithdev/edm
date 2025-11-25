# CLI Specification Changes

## MODIFIED Requirements

### Requirement: Analyze Command
The CLI SHALL provide an `analyze` subcommand for analyzing audio tracks with BPM source attribution.

#### Scenario: Display BPM source in output
- **WHEN** user runs `edm analyze track.mp3`
- **THEN** output shows BPM value with visual indicator of source (metadata/spotify/computed)

#### Scenario: Show timing breakdown
- **WHEN** user runs `edm analyze track.mp3 --verbose`
- **THEN** output shows time spent on each BPM lookup attempt

#### Scenario: JSON output includes BPM source
- **WHEN** user runs `edm analyze track.mp3 --format json`
- **THEN** JSON includes `bpm_source` and `bpm_method` fields

## ADDED Requirements

### Requirement: BPM Source Indicators
The CLI SHALL use visual indicators to show the source of BPM data in table output.

#### Scenario: Metadata source icon
- **WHEN** BPM comes from file metadata
- **THEN** displays "📄" icon or "[M]" prefix in no-color mode

#### Scenario: Spotify source icon
- **WHEN** BPM comes from Spotify API
- **THEN** displays "🎵" icon or "[S]" prefix in no-color mode

#### Scenario: Computed source icon
- **WHEN** BPM is computed by analysis
- **THEN** displays "🔬" icon or "[C]" prefix in no-color mode

#### Scenario: Color coding for sources
- **WHEN** terminal supports colors and `--no-color` is not set
- **THEN** metadata BPM is blue, Spotify BPM is green, computed BPM is yellow

## ADDED Requirements

### Requirement: Offline Mode Flag
The CLI SHALL provide `--offline` flag to skip all network calls during BPM analysis.

#### Scenario: Skip Spotify API in offline mode
- **WHEN** user runs `edm analyze track.mp3 --offline`
- **THEN** BPM lookup tries metadata first, then computes, skipping Spotify API

#### Scenario: Offline mode in batch analysis
- **WHEN** user runs `edm analyze *.mp3 --offline`
- **THEN** no network calls are made for any track

#### Scenario: Offline mode with no metadata
- **WHEN** user runs `edm analyze track.mp3 --offline` and file has no BPM metadata
- **THEN** system computes BPM directly without attempting Spotify lookup

#### Scenario: Combine offline with verbose
- **WHEN** user runs `edm analyze track.mp3 --offline --verbose`
- **THEN** logs show "Skipping Spotify lookup (offline mode)"

## ADDED Requirements

### Requirement: Ignore Metadata Flag
The CLI SHALL provide `--ignore-metadata` flag to skip file metadata lookups during BPM analysis.

#### Scenario: Skip metadata lookup
- **WHEN** user runs `edm analyze track.mp3 --ignore-metadata`
- **THEN** BPM lookup tries Spotify first, then computes, skipping file metadata

#### Scenario: Ignore metadata with offline mode
- **WHEN** user runs `edm analyze track.mp3 --ignore-metadata --offline`
- **THEN** system computes BPM directly (equivalent to --force-compute)

#### Scenario: Ignore metadata in batch analysis
- **WHEN** user runs `edm analyze *.mp3 --ignore-metadata`
- **THEN** no file metadata is read for any track

## ADDED Requirements

### Requirement: Flag Composability
The CLI SHALL support composing flags to achieve fine-grained control over BPM lookup strategy.

#### Scenario: Combine flags for custom strategy
- **WHEN** user runs `edm analyze track.mp3 --ignore-metadata`
- **THEN** strategy is: spotify → computed

#### Scenario: Combine offline and ignore-metadata for forced computation
- **WHEN** user runs `edm analyze track.mp3 --offline --ignore-metadata`
- **THEN** strategy is: computed only

#### Scenario: Only use Spotify (fail if not found)
- **WHEN** user runs `edm analyze track.mp3 --ignore-metadata --no-compute`
- **THEN** only attempts Spotify lookup, fails if track not found

#### Scenario: Show strategy in verbose mode
- **WHEN** user runs `edm analyze track.mp3 --offline --verbose`
- **THEN** logs show "BPM lookup strategy: metadata, computed (offline mode)"

## MODIFIED Requirements

### Requirement: Output Formatting with Rich Library
The CLI SHALL use Rich library to format BPM analysis results with source attribution.

#### Scenario: Table shows BPM source column
- **WHEN** user analyzes multiple tracks in table format
- **THEN** table includes "Source" column showing metadata/spotify/computed

#### Scenario: Timing information includes breakdown
- **WHEN** verbose mode is enabled
- **THEN** timing output shows metadata read time, API call time, computation time separately

## MODIFIED Requirements

### Requirement: Error Handling and User Feedback
The CLI SHALL provide clear error messages for BPM analysis failures at each lookup stage.

#### Scenario: Metadata read error
- **WHEN** file metadata cannot be read
- **THEN** displays warning "Could not read metadata from [file]" and continues to next strategy

#### Scenario: Spotify API error
- **WHEN** Spotify API is unavailable or credentials invalid
- **THEN** displays warning "Spotify lookup failed: [reason]" and falls back to computation

#### Scenario: Spotify credentials not configured
- **WHEN** Spotify lookup is attempted but credentials not set
- **THEN** displays info message "Spotify credentials not configured, skipping API lookup"

#### Scenario: All strategies fail
- **WHEN** metadata, Spotify, and computation all fail
- **THEN** displays error "Failed to determine BPM for [file]" and exits with code 1

## ADDED Requirements

### Requirement: Configuration File Support for BPM
The CLI SHALL support configuring BPM lookup strategy via configuration file.

#### Scenario: Configure lookup strategy in config file
- **WHEN** config file contains `bpm_lookup_strategy = ["spotify", "computed"]`
- **THEN** CLI uses specified strategy, skipping metadata lookup

#### Scenario: Command-line flag overrides config
- **WHEN** config specifies strategy but `--force-compute` flag is used
- **THEN** command-line flag takes precedence

## ADDED Requirements

### Requirement: Batch Analysis Progress
The CLI SHALL show detailed progress for batch BPM analysis indicating current lookup stage.

#### Scenario: Progress shows current lookup stage
- **WHEN** analyzing batch of files
- **THEN** progress bar shows "Analyzing track.mp3 [checking metadata]", "[querying Spotify]", or "[computing]"

#### Scenario: Progress shows success rate
- **WHEN** batch analysis completes
- **THEN** summary shows how many tracks used metadata, Spotify, or computation

#### Scenario: Show failed lookups in summary
- **WHEN** batch analysis completes with some failures
- **THEN** summary shows count of tracks where each lookup stage failed
