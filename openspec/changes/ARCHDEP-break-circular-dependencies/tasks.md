# Tasks

## 1. Create Orchestrator
- [ ] 1.1 Create `src/edm/analysis/orchestrator.py`
- [ ] 1.2 Define `AnalysisOrchestrator` class with DI
- [ ] 1.3 Implement `analyze_full()` method

## 2. Refactor Structure Analysis
- [ ] 2.1 Remove BPM imports from `structure.py`
- [ ] 2.2 Add `bpm` and `downbeat` as parameters
- [ ] 2.3 Update tests to pass explicit parameters

## 3. Update CLI
- [ ] 3.1 Use orchestrator in `cli/commands/analyze.py`
- [ ] 3.2 Verify all analysis types work

## 4. Tests
- [ ] 4.1 Test structure analysis with mocked BPM
- [ ] 4.2 Integration test for full orchestrator
