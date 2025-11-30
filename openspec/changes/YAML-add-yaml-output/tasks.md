## 1. Output Schema
- [x] 1.1 Define `TrackAnalysis` dataclass with new schema structure
- [x] 1.2 Add `to_dict()` method that produces the new format
- [x] 1.3 Update `_analyze_file_impl` to return `TrackAnalysis`

## 2. YAML Output
- [x] 2.1 Add `pyyaml` dependency to `pyproject.toml`
- [x] 2.2 Implement `output_yaml()` function with multi-document support
- [x] 2.3 Add `yaml` to `--format` choices in CLI

## 3. JSON Output
- [x] 3.1 Update `output_json()` to use new schema
- [x] 3.2 Remove redundant fields (`sections` count, flat `bpm_*` fields)

## 4. Testing
- [x] 4.1 Add unit tests for YAML output formatting
- [x] 4.2 Add unit tests for multi-document batch output
- [x] 4.3 Update existing JSON output tests for new schema
