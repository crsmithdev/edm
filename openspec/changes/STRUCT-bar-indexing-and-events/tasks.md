# Tasks: Bar Indexing and Event Format

## Phase 1: Core Data Model Changes

- [x] Add `is_event: bool` field to `DetectedSection` dataclass in `src/edm/analysis/structure_detector.py`
- [x] Update `MSAFDetector._map_to_edm_labels()` to mark drops as events (`is_event=True`)
- [x] Update `EnergyDetector._boundaries_to_sections()` to mark drops as events
- [x] Add `merge_consecutive_other()` function in `src/edm/analysis/structure.py`
- [x] Apply merge function in `analyze_structure()` after `_post_process_sections()`

## Phase 2: Output Format Changes

- [x] Add `format_structure_output()` function in `src/edm/analysis/structure.py` to split spans/events
- [x] Update bar indexing to 1-indexed in `format_structure_output()` (add 1 to all bar numbers)
- [x] Modify `StructureResult` dataclass to include `events: list[tuple[int, str]]` field
- [x] Update CLI output formatter in `src/cli/commands/analyze.py` to display both `structure` and `events`
- [x] Update YAML output format to include separate `events` key

## Phase 2b: Raw Detection Output

- [x] Add `raw` field to `StructureResult` containing original detected sections
- [x] Include in raw: start/end times, start/end bars (fractional), label, confidence
- [x] Update `TrackAnalysis` dataclass with `raw` field
- [x] Update CLI output formatter to include `raw` key in YAML/JSON output
- [x] Preserve raw data before post-processing (merging, 1-indexing)

## Phase 2c: Annotation Template Output

- [x] Add `--annotations` flag to analyze command in `src/cli/main.py`
- [x] When flag is set, switch output format to annotations-only (not alongside standard)
- [x] Template contains: file, duration, bpm, downbeat, time_signature, and `annotations` key
- [x] `annotations` is a list of `[bar, label]` tuples (timestamps derivable from bpm/downbeat)
- [x] Derive initial annotations from detected sections/events as starting point

## Phase 3: Event Detection

- [ ] Add `kick` event detection using `librosa.onset.onset_detect()` for percussion onsets
- [ ] Filter kick events to only include strong onsets (threshold by onset strength)
- [ ] Add kick events to output in `analyze_structure()`

## Phase 4: Tests and Validation

- [x] Update existing structure tests for 1-indexed bar numbers
- [x] Add test for `merge_consecutive_other()` function
- [x] Add test for event vs span separation
- [ ] Add test for kick event detection
- [x] Update integration tests with new output format
- [x] Validate against `data/dj-rhythmcore-narcos_annotated.yaml` (after migrating annotation)

## Phase 5: Evaluation Framework

- [ ] Update `src/edm/evaluation/evaluators/structure.py` to handle event format
- [ ] Add event evaluation metrics (precision/recall for drops, kicks)
- [ ] Update `load_structure_reference()` to parse both span and event formats
- [ ] Test evaluation on annotated data

## Phase 6: Documentation and Migration

- [ ] Migrate existing annotations in `data/` to 1-indexed format
- [ ] Update `STRUCTURE_ANNOTATIONS_GUIDE.md` with new format and conventions
- [ ] Update CLI help text and examples
- [ ] Add CHANGELOG entry documenting breaking changes
- [ ] Update README with example output showing new format
