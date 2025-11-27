# Spec Delta: Performance - Parallel Audio Analysis

## ADDED Requirements

### Requirement: Parallel Processing of Audio Files

The analysis engine SHALL process multiple audio files concurrently to reduce wall-clock time for batch operations.

#### Scenario: User analyzes 50 files with parallel processing

**Given** a directory containing 50 FLAC audio files averaging 5 minutes duration each
**And** the system has 8 CPU cores available
**When** the user runs `edm analyze /path/to/files --workers 8 --offline --ignore-metadata`
**Then** the system shall complete analysis in under 5 minutes
**And** all 50 files shall have valid BPM results
**And** the results shall match sequential processing output

#### Scenario: User evaluates BPM accuracy on 50 files with parallel workers

**Given** a directory with 50 audio files with metadata BPM tags
**When** the user runs `edm evaluate bpm --source /path/to/files --reference metadata --sample-size 50 --workers 8`
**Then** the system shall compute BPM for all files in parallel
**And** the evaluation metrics (MAE, RMSE, accuracy) shall match sequential execution
**And** the total execution time shall be reduced by at least 5x compared to sequential

### Requirement: Worker Count Control via CLI Flag

The system SHALL provide a --workers CLI flag to give users explicit control over the degree of parallelism to balance performance and resource usage.

#### Scenario: User specifies custom worker count

**Given** the user wants to analyze files with 4 parallel workers
**When** they run `edm analyze track1.mp3 track2.mp3 track3.mp3 track4.mp3 --workers 4`
**Then** the system shall spawn 4 worker processes
**And** distribute files across workers for concurrent processing
**And** display progress updates as files complete

#### Scenario: User forces sequential processing

**Given** the user wants deterministic sequential execution
**When** they run `edm analyze files/*.flac --workers 1`
**Then** the system shall process files one at a time in order
**And** the behavior shall be identical to the pre-parallelization implementation

#### Scenario: User requests invalid worker count

**Given** the user specifies `--workers 0`
**When** the command is executed
**Then** the system shall display an error message "Worker count must be at least 1"
**And** exit with non-zero status code

**Given** the user specifies `--workers 100` on a system with 8 cores
**When** the command is executed
**Then** the system shall display a warning "Worker count (100) exceeds CPU count (8)"
**And** proceed with 100 workers (allowing user override for specific use cases)

### Requirement: Result Ordering Preservation

The system SHALL return results in the same order as input files, regardless of which worker completes first.

#### Scenario: Parallel workers complete out of order

**Given** three files: fast.mp3 (2min), medium.mp3 (4min), slow.mp3 (8min)
**And** parallel processing with 3 workers
**When** workers complete in order: fast → medium → slow
**Then** the results array shall contain results in order [fast, medium, slow]
**And** the output table shall display files in original order

### Requirement: Graceful Error Handling in Parallel Mode

The system SHALL ensure that a failure in one file's analysis does not prevent processing of other files or cause system instability.

#### Scenario: One file fails during parallel analysis

**Given** 10 audio files where file #5 is corrupted
**When** analyzing with `--workers 4`
**Then** the system shall continue processing files 1-4, 6-10
**And** file #5 shall be marked as failed with error message in results
**And** 9 successful results shall be returned
**And** the exit code shall be 0 (partial success)

#### Scenario: Multiple files fail simultaneously

**Given** 20 files where files #3, #7, #15 are corrupted
**When** analyzing with `--workers 8`
**Then** all 3 failures shall be logged independently
**And** 17 successful results shall be returned
**And** the error count shall be displayed in the summary

### Requirement: Real-Time Progress Updates for Parallel Execution

The system SHALL provide real-time progress updates so users see progress as files complete, even when processing happens across multiple workers.

#### Scenario: Progress bar updates during parallel processing

**Given** 50 files being analyzed with 8 workers
**When** workers complete files asynchronously
**Then** the progress bar shall update immediately as each file finishes
**And** the progress percentage shall reflect actual completed files
**And** the display shall not flicker or show incorrect counts

### Requirement: Graceful Shutdown on Interruption

The system SHALL support graceful shutdown so users can cancel parallel operations cleanly without leaving zombie processes.

#### Scenario: User interrupts parallel processing with Ctrl+C

**Given** parallel analysis is running with 8 workers processing 100 files
**And** 30 files have completed
**When** the user presses Ctrl+C (sends SIGINT)
**Then** the system shall immediately stop accepting new work
**And** terminate all worker processes within 5 seconds
**And** display "Operation cancelled by user"
**And** exit with status code 130 (standard SIGINT exit code)

#### Scenario: System sends SIGTERM to running evaluation

**Given** a BPM evaluation running with 4 workers
**When** the process receives SIGTERM
**Then** the system shall save partial results to disk
**And** terminate all workers gracefully
**And** exit within 10 seconds

### Requirement: Sequential Processing Default for Backward Compatibility

The system SHALL default to sequential processing so existing workflows continue to work without modification.

#### Scenario: User runs analyze command without --workers flag

**Given** the user runs `edm analyze track.mp3` (no --workers specified)
**When** the command executes
**Then** the system shall process files sequentially (workers=1)
**And** the behavior shall be identical to previous versions

#### Scenario: Existing scripts use analyze without parallelization

**Given** an automated CI pipeline running `edm analyze test-fixtures/*.flac`
**When** the pipeline runs after upgrading to the parallel version
**Then** the pipeline shall continue to work without changes
**And** the output format shall remain the same

## MODIFIED Requirements

None - this change adds new functionality without modifying existing requirements.

## REMOVED Requirements

None - sequential processing remains fully supported.
