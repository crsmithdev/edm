## 1. Data Model
- [x] 1.1 Define event labels list (e.g., `drop`) vs span labels
- [x] 1.2 Update output formatting to emit 2-element tuples for event labels

## 2. Output
- [x] 2.1 Update `_analyze_file_impl` to format drops as `[bar, label]`
- [x] 2.2 Update YAML/JSON output tests

## 3. Evaluation
- [x] 3.1 Update reference loader to parse 2-element tuples as events
- [x] 3.2 Adjust evaluation to handle event vs span comparison

## 4. Testing
- [x] 4.1 Update structure output tests for polymorphic format
- [x] 4.2 Update annotation fixtures
