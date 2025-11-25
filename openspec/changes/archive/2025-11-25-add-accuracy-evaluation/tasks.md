# Implementation Tasks: Add Accuracy Evaluation Framework

## Phase 1: Library Module Structure

### Setup
- [x] Create `src/edm/evaluation/` directory
- [x] Create `src/edm/evaluation/__init__.py`
- [x] Create `src/edm/evaluation/common.py`
- [x] Create `src/edm/evaluation/reference.py`
- [x] Create `src/edm/evaluation/evaluators/` directory
- [x] Create `src/edm/evaluation/evaluators/__init__.py`
- [x] Create `src/edm/evaluation/evaluators/bpm.py`
- [x] Create `benchmarks/results/` directory structure
- [x] Create `benchmarks/results/accuracy/bpm/` subdirectory
- [x] Create `tests/fixtures/reference/` subdirectory

### Shared Infrastructure (`src/edm/evaluation/common.py`)
- [x] Implement `discover_audio_files()` - find audio files recursively
- [x] Implement `AudioFileCollection` class for file management
- [x] Implement sampling (random with seed, full dataset)
- [x] Implement metrics calculation (MAE, RMSE, accuracy within tolerance)
- [x] Implement error distribution calculation
- [x] Implement outlier identification
- [x] Implement result storage (JSON + Markdown)
- [x] Implement git commit/branch tracking
- [x] Implement symlink creation for latest results
- [x] Add structlog logging integration

### Reference Loading (`src/edm/evaluation/reference.py`)
- [x] Implement `load_reference()` with auto-detection
- [x] Implement CSV reference loading
- [x] Implement JSON reference loading
- [x] Implement Spotify API reference loading
- [x] Implement metadata (ID3 tags) reference loading
- [x] Handle missing files gracefully
- [x] Validate analysis-specific reference support

## Phase 2: BPM Evaluator

### BPM Evaluation Logic (`src/edm/evaluation/evaluators/bpm.py`)
- [x] Implement `BPMEvaluator` class
- [x] Implement `evaluate()` method
- [x] Discover and sample audio files
- [x] Load reference data
- [x] Compute BPM using `edm.analysis.bpm`
- [x] Compare with reference and calculate errors
- [x] Calculate summary metrics
- [x] Identify octave errors (halving/doubling)
- [x] Generate tempo-stratified breakdown
- [x] Save results (JSON + Markdown)
- [x] Handle errors gracefully, continue on failures

## Phase 3: CLI Integration

### CLI Command (`src/cli/commands/evaluate.py`)
- [x] Create `edm evaluate` command group
- [x] Implement `edm evaluate bpm` subcommand
- [x] Add `--source` argument (required)
- [x] Add `--reference` argument (spotify, metadata, or file path)
- [x] Add `--sample-size` argument (default: 100)
- [x] Add `--full` flag for all files
- [x] Add `--seed` argument for reproducibility
- [x] Add `--tolerance` argument (default: 2.5)
- [x] Add `--output` argument for results directory
- [x] Display progress with Rich console
- [x] Print summary after evaluation

## Phase 4: Reference Data

### Reference Setup
- [x] Create `tests/fixtures/reference/README.md`
- [x] Document CSV format
- [x] Document JSON format
- [x] Document Spotify and metadata reference sources

## Phase 5: Results Documentation

### Results Format
- [x] Create `benchmarks/results/README.md`
- [x] Document JSON schema
- [x] Document Markdown format
- [x] Explain git commit tracking
- [x] Provide usage examples

## Phase 6: Testing

### Unit Tests
- [x] Create `tests/unit/test_evaluation/` directory
- [x] Create `tests/unit/test_evaluation/test_common.py`
- [x] Create `tests/unit/test_evaluation/test_reference.py`
- [x] Create `tests/unit/test_evaluation/test_bpm.py`
- [x] Test file discovery
- [x] Test sampling (random with seed, full)
- [x] Test reference loading (CSV, JSON, metadata, spotify)
- [x] Test metrics calculation
- [x] Test result saving
- [x] Test BPM evaluation with mocked analysis

## Phase 7: Documentation

### Usage Documentation
- [x] Add "Accuracy Evaluation" section to README.md
- [x] Document CLI usage examples
- [x] Document reference source options
- [x] Document output format and location
- [x] Add docstrings to all public functions

## Phase 8: Quality Assurance

### Code Quality
- [x] Run linter (ruff)
- [x] Run formatter (black)
- [x] Run tests (pytest)
- [x] Manual testing with real music files
- [x] Validate results format (JSON + Markdown)
- [x] Verify git commit tracking works
