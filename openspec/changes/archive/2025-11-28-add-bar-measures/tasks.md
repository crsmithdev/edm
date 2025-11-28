# Tasks: Add Bar/Measure Calculation

## 1. Core Bar Utilities

- [x] 1.1 Create `src/edm/analysis/bars.py` module
- [x] 1.2 Implement `time_to_bars(time_seconds, bpm, time_signature)` function
- [x] 1.3 Implement `bars_to_time(bar, bpm, time_signature)` function
- [x] 1.4 Implement `bar_count_for_range(start_time, end_time, bpm, time_signature)` function
- [x] 1.5 Add optional `beat_grid` parameter to all functions for future extensibility
- [x] 1.6 Handle edge cases (None BPM, invalid time signature, negative values)

## 2. Data Models

- [x] 2.1 Add bar fields to `DetectedSection` dataclass (`start_bar`, `end_bar`, `bar_count`)
- [x] 2.2 Update `StructureResult` model to include BPM reference
- [x] 2.3 Add `TimeSignature` type alias (tuple[int, int] defaulting to (4, 4))
- [x] 2.4 Ensure backward compatibility (bar fields Optional, default None)

## 3. Structure Analysis Integration

- [x] 3.1 Update `analyze_structure()` to call BPM analysis if not provided
- [x] 3.2 Calculate bar positions for each detected section
- [x] 3.3 Add `include_bars: bool = True` parameter to control bar calculation
- [x] 3.4 Handle cases where BPM unavailable (bar fields remain None)
- [x] 3.5 Update structure detector classes to accept BPM parameter

## 4. CLI Output

- [x] 4.1 Update structure table display to show bar counts
- [x] 4.2 Add bar column to structure section tables (e.g., "16 bars")
- [x] 4.3 Update JSON output format to include bar fields
- [x] 4.4 Add `--no-bars` flag to disable bar calculations if needed

## 5. Evaluation Support

- [x] 5.1 Update structure evaluator to support bar-based ground truth
- [x] 5.2 Allow CSV annotations with `start_bar`/`end_bar` columns instead of `start`/`end`
- [x] 5.3 Add BPM column or metadata support for bar-to-time conversion
- [x] 5.4 Support mixed format (prefer bars if available, fall back to time)
- [x] 5.5 Convert bar annotations to time using track BPM before evaluation
- [x] 5.6 Add bar boundary alignment metrics (±0.5 bar tolerance)
- [x] 5.7 Report section length accuracy in bars vs time
- [x] 5.8 Update annotation CSV example/documentation with bar format

## 6. Testing

- [x] 6.1 Unit tests for bar conversion functions
- [x] 6.2 Test time signature variations (4/4, 3/4, 6/8)
- [x] 6.3 Test edge cases (no BPM, fractional bars, rounding)
- [x] 6.4 Integration test: structure analysis with bar counts
- [x] 6.5 Test bar queries (section at bar N)
- [x] 6.6 Test backward compatibility (existing code without bars)
- [x] 6.7 Test bar-based annotation loading from CSV
- [x] 6.8 Test mixed format annotations (time + bar columns)
- [x] 6.9 Test bar-to-time conversion for evaluation

## 7. Documentation

- [x] 7.1 Update `docs/cli-reference.md` with bar display examples
- [x] 7.2 Document bar-based annotation CSV format (with BPM metadata)
- [x] 7.3 Add example bar-based annotation file to fixtures
- [x] 7.4 Update `docs/architecture.md` with bar calculation design
- [x] 7.5 Add docstrings to all bar utility functions
- [x] 7.6 Document future beat grid integration approach
- [x] 7.7 Add examples of bar-based structure representation
- [x] 7.8 Document annotation workflow (listening + marking bars)

## Dependencies

- Tasks 2.x depend on 1.x (models use bar utilities)
- Tasks 3.x depend on both 1.x and 2.x
- Tasks 4.x and 5.x depend on 3.x
- Task 6.x runs throughout
- Task 7.x runs after implementation complete

## Validation Criteria

- [x] `openspec validate add-bar-measures --strict` passes
- [x] All tests pass: `uv run pytest tests/`
- [x] Bar calculations accurate for standard EDM BPMs (120-150)
- [x] Structure output includes bar counts when BPM available
- [x] Bar fields are None when BPM unavailable (graceful degradation)
- [x] Code designed for beat grid integration (documented extension points)
