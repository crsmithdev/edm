# Core Library Specification Changes - Accuracy Evaluation

## ADDED Requirements

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
