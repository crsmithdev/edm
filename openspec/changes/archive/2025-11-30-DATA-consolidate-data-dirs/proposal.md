# Change: Consolidate Data Directories

## Why
Annotations and accuracy reports are scattered across `/tests/fixtures/reference/` and `/benchmarks/results/accuracy/`. A unified `data/` directory improves discoverability and follows Python conventions.

## What Changes
- **BREAKING**: Move `tests/fixtures/reference/` → `data/annotations/`
- **BREAKING**: Move `benchmarks/results/accuracy/` → `data/accuracy/`
- Update all code to read/write from new locations by default
- Remove empty legacy directories

## Impact
- Affected specs: `cli`
- Affected code:
  - `src/edm/evaluation/evaluators/bpm.py` (default output path)
  - `src/edm/evaluation/evaluators/structure.py` (default output path, reference loading)
  - `src/edm/evaluation/reference.py` (reference loading)
  - `src/cli/commands/evaluate.py` (CLI help text)
  - Tests referencing fixture paths
