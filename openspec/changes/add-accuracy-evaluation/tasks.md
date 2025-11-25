# Implementation Tasks: Add Accuracy Evaluation Framework

## Phase 1: Directory Structure and Shared Infrastructure

### Setup
- [ ] Create `benchmarks/` directory
- [ ] Create `benchmarks/accuracy/` subdirectory
- [ ] Create `tests/fixtures/reference/` subdirectory
- [ ] Create `benchmarks/results/` directory
- [ ] Create `benchmarks/results/accuracy/` subdirectory
- [ ] Create `benchmarks/results/accuracy/bpm/` subdirectory

### Shared Infrastructure (`benchmarks/accuracy/common.py`)
- [ ] Implement `discover_audio_files(source_path: Path) -> List[Path]`
  - [ ] Find all audio files recursively (.mp3, .flac, .wav, .m4a)
  - [ ] Filter by supported formats
  - [ ] Return sorted list of paths

- [ ] Implement sampling functions
  - [ ] `sample_random(files: List[Path], size: int, seed: Optional[int]) -> List[Path]`
  - [ ] `sample_full(files: List[Path]) -> List[Path]`

- [ ] Implement reference loading
  - [ ] `load_reference_auto(reference_arg: str, analysis_type: str, source_path: Path, value_field: str) -> Dict[Path, Any]`
  - [ ] `load_reference_csv(path: Path, value_field: str) -> Dict[Path, float]`
  - [ ] `load_reference_json(path: Path, value_field: str) -> Dict[Path, float]`
  - [ ] `load_spotify_reference(source_path: Path) -> Dict[Path, float]` - BPM only
  - [ ] `load_metadata_reference(source_path: Path, value_field: str) -> Dict[Path, Any]` - BPM, key
  - [ ] Handle missing files gracefully
  - [ ] Validate analysis-specific reference support (raise error if unsupported)
  
- [ ] Implement metrics calculation
  - [ ] `calculate_mae(errors: List[float]) -> float`
  - [ ] `calculate_rmse(errors: List[float]) -> float`
  - [ ] `calculate_accuracy_within_tolerance(errors: List[float], tolerance: float) -> float`
  - [ ] `calculate_error_distribution(errors: List[float], bins: int) -> Dict[str, int]`
  - [ ] `identify_outliers(results: List[dict], n: int) -> List[dict]`

- [ ] Implement result storage
  - [ ] `save_results_json(results: dict, output_path: Path)`
  - [ ] `save_results_markdown(results: dict, output_path: Path)`
  - [ ] `save_error_distribution_plot(errors: List[float], output_path: Path)` - Optional matplotlib visualization
  - [ ] `get_git_commit() -> str` - Get current git commit hash
  - [ ] `get_git_branch() -> str` - Get current git branch
  - [ ] `create_symlinks(output_path: Path)` - Create latest.json and latest.md symlinks

- [ ] Add logging setup
  - [ ] Configure structlog for script output
  - [ ] Add progress indicators (simple print statements or tqdm)

## Phase 2: BPM Evaluator

### BPM Evaluation Logic (`benchmarks/accuracy/bpm.py`)
- [ ] Implement `evaluate_bpm(args)`
  - [ ] Parse arguments
  - [ ] Discover audio files
  - [ ] Sample files according to strategy
  - [ ] Load reference using `common.load_reference_auto()` with analysis_type="bpm"
  - [ ] Iterate through sampled files
    - [ ] Compute BPM using `edm.analysis.bpm.analyze_bpm(force_compute=True, offline=True)`
    - [ ] Compare with reference
    - [ ] Record result (file, reference, computed, error, success, time)
    - [ ] Log progress
  - [ ] Calculate metrics
  - [ ] Save results (JSON + Markdown)
  - [ ] Save error distribution plot (optional matplotlib)
  - [ ] Create symlinks
  - [ ] Print summary

- [ ] Handle errors gracefully
  - [ ] Catch computation errors
  - [ ] Continue processing remaining files
  - [ ] Log failures with details

## Phase 3: Main Script

### Main Evaluation Script (`benchmarks/accuracy/evaluate.py`)
- [ ] Import argparse and submodules
- [ ] Create argument parser
  - [ ] Add `bpm` subcommand
    - [ ] `--source` (required): Path to music directory
    - [ ] `--sample-size` (default: 100): Number of files to sample
    - [ ] `--reference` (required): Reference source ('spotify', 'metadata', or path to CSV/JSON file)
    - [ ] `--output` (optional): Output directory (default: benchmarks/results/accuracy/bpm/)
    - [ ] `--seed` (optional): Random seed for reproducibility
    - [ ] `--full`: Use all files (ignore sample-size)
    - [ ] `--tolerance` (default: 2.5): BPM tolerance for accuracy
  - [ ] Add `drops` subcommand placeholder (future)
  
- [ ] Implement main() function
  - [ ] Parse arguments
  - [ ] Validate inputs (paths exist, sample size positive, etc.)
  - [ ] Route to appropriate evaluator (bpm, drops, etc.)
  
- [ ] Add `if __name__ == "__main__"` block

## Phase 4: Reference Data

### Initial Reference Setup
- [ ] Create `tests/fixtures/reference/README.md`
  - [ ] Document CSV format
  - [ ] Document JSON format
  - [ ] Provide examples

- [ ] Create `tests/fixtures/reference/bpm_tagged.csv` (or use existing test fixtures)
  - [ ] Use existing test fixtures as starting point
  - [ ] Add more manually tagged files (optional)

- [ ] Create example reference file
  - [ ] Include at least 10-20 files for initial testing

## Phase 5: Results Documentation

### Results Format Documentation
- [ ] Create `benchmarks/results/README.md`
  - [ ] Document JSON schema
  - [ ] Document Markdown format
  - [ ] Explain git commit tracking
  - [ ] Provide usage examples for AI assistants

- [ ] Add `.gitignore` entries
  - [ ] Ignore `benchmarks/results/**/*.json` (or decide to track them)
  - [ ] Ignore `benchmarks/results/**/*.md`
  - [ ] Keep structure files

## Phase 6: Testing

### Unit Tests
- [ ] Create `tests/unit/test_accuracy_common.py`
  - [ ] Test file discovery
  - [ ] Test random sampling (with seed)
  - [ ] Test full sampling
  - [ ] Test reference loading (CSV/JSON)
  - [ ] Test `load_reference_auto()` with different input types
  - [ ] Test Spotify reference loading (mock SpotifyClient)
  - [ ] Test metadata reference loading (mock read_metadata)
  - [ ] Test analysis-specific reference validation (e.g., spotify only for bpm, metadata for bpm/key)
  - [ ] Test metrics calculation (MAE, RMSE, accuracy, distribution, outliers)
  - [ ] Test result saving (JSON/Markdown)
  - [ ] Test git info retrieval

- [ ] Create `tests/unit/test_accuracy_bpm.py`
  - [ ] Test BPM evaluation with synthetic audio
  - [ ] Test error handling
  - [ ] Mock `analyze_bpm` for controlled tests

### Integration Tests
- [ ] Create `tests/integration/test_evaluate_script.py`
  - [ ] Test running BPM evaluation end-to-end
  - [ ] Test with different sample sizes
  - [ ] Test with different sampling strategies
  - [ ] Test with test fixtures as ground truth
  - [ ] Test output file generation
  - [ ] Test symlink creation
  - [ ] Verify JSON schema
  - [ ] Verify Markdown format

### Test Fixtures
- [ ] Create `tests/fixtures/reference_bpm.csv`
  - [ ] Add reference for existing test audio files
- [ ] Ensure test audio files have known BPM values

## Phase 7: Documentation

### Usage Documentation
- [ ] Update main `README.md`
  - [ ] Add "Accuracy Evaluation" section (developer-focused)
  - [ ] Add basic usage examples
  - [ ] Link to scripts/reference/README.md

- [ ] Add docstrings
  - [ ] All functions in `scripts/accuracy/common.py`
  - [ ] All functions in `scripts/accuracy/bpm.py`
  - [ ] Main script help text

- [ ] Create example workflow
  - [ ] How to create reference file
  - [ ] How to run evaluation
  - [ ] How to interpret results
  - [ ] How to compare across commits

## Phase 8: Quality Assurance

### Code Quality
- [ ] Run linter: `ruff check benchmarks/`
- [ ] Run formatter: `black benchmarks/`
- [ ] Run type checker: `mypy benchmarks/` (if type hints added)

### Testing
- [ ] Run full test suite: `pytest`
- [ ] Check test coverage for new code
- [ ] Manual testing with real music files
  - [ ] Test with 10 files
  - [ ] Test with 100 files
  - [ ] Test with full dataset (if available)

### Validation
- [ ] Validate with OpenSpec: `openspec validate add-accuracy-evaluation --strict`
- [ ] Verify results format
  - [ ] JSON is valid and parseable
  - [ ] Markdown is well-formatted
  - [ ] Symlinks are created correctly
  - [ ] Git commit hash is captured

### AI Assistant Testing
- [ ] Test Claude's ability to read latest.md
- [ ] Test Claude's ability to compare results across commits
- [ ] Test Claude's ability to identify regressions
- [ ] Verify JSON parsing works smoothly

## Phase 9: Future Enhancements (Optional)

- [ ] Add advanced visualization support (matplotlib)
  - [ ] Accuracy trend over time (comparing multiple evaluations)
  - [ ] Per-genre breakdown plots
- [ ] Add more analysis types `
  - [ ] Drop detection evaluator (`benchmarks/accuracy/drops.py`)
  - [ ] Key detection evaluator (`benchmarks/accuracy/key.py`)
- [ ] Add Spotify API caching
  - [ ] Cache Spotify lookups to `benchmarks/results/accuracy/.cache/spotify_lookups.json`
  - [ ] Include timestamp and invalidation logic
- [ ] Add more reference sources
  - [ ] Beatport API (if available)
  - [ ] File metadata (ID3 tags) as reference
- [ ] Add stratified sampling
- [ ] Add CI/CD integration (run on commits, fail if accuracy drops)
- [ ] Store results in git for historical tracking

## Estimated Time

- Phase 1: ~2-3 hours (shared infrastructure)
- Phase 2: ~2-3 hours (BPM evaluator)
- Phase 3: ~1 hour (main script)
- Phase 4: ~1 hour (ground truth setup)
- Phase 5: ~30 minutes (results documentation)
- Phase 6: ~2-3 hours (testing)
- Phase 7: ~1-2 hours (documentation)
- Phase 8: ~1 hour (QA)
- **Total**: ~10-15 hours

## Notes

- This approach is simpler than full CLI integration, focusing on internal tooling
- No need for complex plugin architecture or public API
- Easy to iterate and experiment
- Can be extended as needed for new analysis types
- Optimized for AI assistant integration (Markdown + JSON + git tracking)
