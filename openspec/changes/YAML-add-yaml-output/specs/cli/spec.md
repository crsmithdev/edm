## ADDED Requirements

### Requirement: YAML Output Format
The CLI SHALL support YAML as an output format for analysis results.

#### Scenario: Single track YAML output
- **WHEN** user runs `edm analyze track.flac --format yaml`
- **THEN** outputs analysis results as a single YAML document to stdout

#### Scenario: Batch YAML output
- **WHEN** user runs `edm analyze *.flac --format yaml`
- **THEN** outputs multi-document YAML with each track separated by `---`

#### Scenario: YAML to file
- **WHEN** user runs `edm analyze *.flac --format yaml -o results.yaml`
- **THEN** writes multi-document YAML to the specified file

#### Scenario: Appendable output
- **WHEN** user runs `edm analyze newtrack.flac --format yaml >> results.yaml`
- **THEN** appended YAML documents are valid when concatenated with existing file

## MODIFIED Requirements

### Requirement: Analyze Command
The CLI SHALL provide an `analyze` subcommand for analyzing audio tracks with a `--types` flag to specify which analyses to perform and a `--format` flag supporting `table`, `json`, and `yaml` output formats.

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
- **THEN** writes analysis results to the specified file in the format matching the file extension or `--format` flag

#### Scenario: JSON output format
- **WHEN** user runs `edm analyze track.mp3 --format json`
- **THEN** displays results as JSON to stdout with hierarchical structure grouping related fields

#### Scenario: YAML output format
- **WHEN** user runs `edm analyze track.mp3 --format yaml`
- **THEN** displays results as YAML to stdout

#### Scenario: Invalid input file
- **WHEN** user runs `edm analyze nonexistent.mp3`
- **THEN** displays error message and exits with non-zero status code

#### Scenario: Invalid analysis type
- **WHEN** user runs `edm analyze track.mp3 --types invalid`
- **THEN** displays error listing valid analysis types and exits with non-zero status code

### Requirement: Structured Output Schema
The CLI SHALL output analysis results in a hierarchical schema that groups related fields and uses bar-based structure sections.

#### Scenario: Tempo fields grouped
- **WHEN** user requests JSON or YAML output
- **THEN** BPM, downbeat, and time signature appear under a `tempo` key

#### Scenario: Structure as compact tuples
- **WHEN** user requests JSON or YAML output with structure analysis
- **THEN** structure sections appear as `[start_bar, end_bar, label]` arrays

#### Scenario: Bar times computable
- **WHEN** output includes `tempo.bpm`, `tempo.downbeat`, and `tempo.time_signature`
- **THEN** bar timestamps can be computed as `downbeat + (bar - 1) * (60 * beats_per_bar / bpm)`
