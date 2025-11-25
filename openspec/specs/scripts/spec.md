# scripts Specification

## Purpose
TBD - created by archiving change add-accuracy-evaluation. Update Purpose after archive.
## Requirements
### Requirement: BPM Accuracy Evaluation Script
The project SHALL provide an internal Python script for evaluating BPM detection accuracy against ground truth data.

#### Scenario: Run BPM evaluation with ground truth file
- **WHEN** developer runs `python scripts/evaluate.py bpm --source /path/to/music --ground-truth scripts/ground_truth/bpm_tagged.csv`
- **THEN** system evaluates sampled files, computes BPM, compares with ground truth, and saves results in JSON and Markdown formats

#### Scenario: Specify sample size
- **WHEN** developer runs with `--sample-size 50`
- **THEN** system randomly samples exactly 50 files from source directory

#### Scenario: Evaluate full dataset
- **WHEN** developer runs with `--full` flag
- **THEN** system evaluates all audio files in source directory regardless of sample size

#### Scenario: Reproducible sampling
- **WHEN** developer runs with `--seed 42`
- **THEN** system produces identical sample on repeated runs with same seed

#### Scenario: Custom tolerance
- **WHEN** developer runs with `--tolerance 5.0`
- **THEN** system uses ±5.0 BPM as tolerance for accuracy calculation

#### Scenario: Auto-generate output paths
- **WHEN** developer runs without --output option
- **THEN** system generates timestamped JSON and Markdown files in `results/bpm/` with git commit hash

### Requirement: Ground Truth Loading
The script SHALL support loading ground truth data from CSV and JSON files with flexible field mapping.

#### Scenario: Load ground truth from CSV
- **WHEN** script loads CSV file with columns path,bpm
- **THEN** system creates mapping from file paths to BPM values

#### Scenario: Load ground truth from JSON
- **WHEN** script loads JSON file with array of objects containing path and bpm fields
- **THEN** system creates mapping from file paths to BPM values

#### Scenario: Handle missing files gracefully
- **WHEN** ground truth contains paths that don't exist
- **THEN** system logs warning and skips those entries without failing

#### Scenario: Validate ground truth file format
- **WHEN** ground truth file has unsupported extension
- **THEN** system displays error message and exits with code 1

### Requirement: Sampling Strategies
The script SHALL support multiple sampling strategies for file selection.

#### Scenario: Random sampling with seed
- **WHEN** using random sampling strategy with seed
- **THEN** system uses Python's random.sample with provided seed for reproducibility

#### Scenario: Full dataset sampling
- **WHEN** using full sampling strategy
- **THEN** system returns all discovered audio files ignoring sample size parameter

#### Scenario: Handle sample size larger than available files
- **WHEN** requested sample size exceeds available files
- **THEN** system logs warning and uses all available files

### Requirement: Metrics Calculation
The script SHALL compute standard accuracy metrics for comparing computed values against ground truth.

#### Scenario: Calculate Mean Absolute Error
- **WHEN** computing metrics from evaluation results
- **THEN** system calculates MAE as mean of absolute differences

#### Scenario: Calculate Root Mean Square Error
- **WHEN** computing metrics from evaluation results
- **THEN** system calculates RMSE as square root of mean squared errors

#### Scenario: Calculate accuracy within tolerance
- **WHEN** computing metrics with tolerance threshold
- **THEN** system calculates percentage of results within tolerance

#### Scenario: Generate error distribution
- **WHEN** computing metrics from evaluation results
- **THEN** system generates histogram of error distribution with bins

#### Scenario: Identify worst outliers
- **WHEN** computing metrics
- **THEN** system identifies and returns top 10 results with largest absolute errors

### Requirement: Result Persistence
The script SHALL save evaluation results in both JSON and Markdown formats with git tracking.

#### Scenario: Save JSON results
- **WHEN** evaluation completes
- **THEN** system saves detailed results to JSON file with metadata, summary, and individual file results

#### Scenario: Save Markdown summary
- **WHEN** evaluation completes
- **THEN** system saves human-readable summary to Markdown file with metrics table and outliers

#### Scenario: Track git commit
- **WHEN** saving results
- **THEN** system captures current git commit hash and branch name in metadata

#### Scenario: Create symlinks to latest results
- **WHEN** saving results
- **THEN** system creates `latest.json` and `latest.md` symlinks pointing to most recent results

#### Scenario: Generate timestamped filenames
- **WHEN** no output path specified
- **THEN** system generates filename pattern `YYYY-MM-DD_bpm_eval_commit-{hash}.json`

### Requirement: File Discovery
The script SHALL discover audio files recursively from source directory.

#### Scenario: Discover supported audio formats
- **WHEN** script scans source directory
- **THEN** system finds files with extensions .mp3, .flac, .wav, .m4a recursively

#### Scenario: Return sorted file list
- **WHEN** files are discovered
- **THEN** system returns alphabetically sorted list of absolute paths

#### Scenario: Handle empty directory
- **WHEN** source directory contains no audio files
- **THEN** system displays error message and exits with code 1

### Requirement: Progress Tracking
The script SHALL provide progress feedback during evaluation.

#### Scenario: Log evaluation progress
- **WHEN** processing files
- **THEN** system logs file number, filename, and result at INFO level

#### Scenario: Show completion summary
- **WHEN** evaluation completes
- **THEN** system prints summary with total files, successful evaluations, MAE, RMSE, accuracy

#### Scenario: Log errors without stopping
- **WHEN** computation fails for a file
- **THEN** system logs error with file path and exception, continues with remaining files

### Requirement: Error Handling
The script SHALL handle errors gracefully with clear messages.

#### Scenario: Handle invalid source path
- **WHEN** source path does not exist
- **THEN** system displays error message and exits with code 1

#### Scenario: Handle missing ground truth file
- **WHEN** ground truth file does not exist
- **THEN** system displays error message and exits with code 1

#### Scenario: Handle computation errors
- **WHEN** BPM computation fails for a file
- **THEN** system marks result as failed, logs error, continues processing

#### Scenario: Handle all failures
- **WHEN** all file computations fail
- **THEN** system displays error message and exits with code 1

### Requirement: Extensibility for Multiple Analysis Types
The script SHALL support multiple analysis types through subcommands.

#### Scenario: BPM subcommand
- **WHEN** developer runs `python scripts/evaluate.py bpm`
- **THEN** system uses BPM-specific evaluation logic from `scripts/accuracy/bpm.py`

#### Scenario: Future drops subcommand
- **WHEN** developer runs `python scripts/evaluate.py drops` (future)
- **THEN** system uses drop detection evaluation logic from `scripts/accuracy/drops.py`

#### Scenario: Display available subcommands
- **WHEN** developer runs `python scripts/evaluate.py --help`
- **THEN** system displays list of available analysis types

### Requirement: AI Assistant Integration
Results SHALL be formatted for easy parsing and comparison by AI assistants.

#### Scenario: Human-readable Markdown summary
- **WHEN** AI assistant reads `results/bpm/latest.md`
- **THEN** content is well-formatted Markdown with clear metrics, outliers table, and error distribution

#### Scenario: Machine-parseable JSON
- **WHEN** AI assistant reads `results/bpm/latest.json`
- **THEN** content is valid JSON with clear structure: metadata, summary, outliers, individual results

#### Scenario: Git commit tracking for comparison
- **WHEN** AI assistant compares results across commits
- **THEN** metadata includes git_commit and git_branch fields for version tracking

#### Scenario: Symlinks for latest results
- **WHEN** AI assistant needs most recent results
- **THEN** can reliably use `results/bpm/latest.md` or `latest.json` symlinks

