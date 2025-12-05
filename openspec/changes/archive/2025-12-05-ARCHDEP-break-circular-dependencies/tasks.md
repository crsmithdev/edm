# Tasks

## 1. Create Orchestrator
- [x] 1.1 Create `src/edm/analysis/orchestrator.py`
- [x] 1.2 Define `AnalysisOrchestrator` class with DI
- [x] 1.3 Implement `analyze_full()` method

## 2. Refactor Structure Analysis
- [x] 2.1 Remove BPM imports from `structure.py`
- [x] 2.2 Add `bpm` and `downbeat` as parameters
- [x] 2.3 Update tests to pass explicit parameters

## 3. Update CLI
- [x] 3.1 Use orchestrator in `cli/commands/analyze.py`
- [x] 3.2 Verify all analysis types work

## 4. Tests
- [x] 4.1 Test structure analysis with mocked BPM
- [x] 4.2 Integration test for full orchestrator
