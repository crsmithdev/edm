# Change: Add Bar/Measure Calculation to Analysis

## Why

Structure sections are currently expressed only in time (seconds), which is less meaningful for music analysis. Musicians and DJs think in bars/measures (typically 4 beats = 1 bar in EDM). Expressing structure as "16 bars" is more intuitive than "30.5 seconds."

This lays groundwork for future beat grid implementation while providing immediate utility for structure analysis.

## What Changes

- Add bar/measure calculation based on detected BPM
- Include bar counts in structure analysis results
- Design time-to-bar conversion utilities that will integrate with future beat grid
- Update CLI output to display bar counts alongside time
- Add bar-based query utilities (e.g., "what section is bar 64 in?")

## Impact

- Affected specs: `analysis`
- Affected code:
  - `src/edm/analysis/structure.py` - Add bar calculations to structure results
  - `src/edm/models/base.py` - Add bar fields to data models
  - `src/cli/commands/analyze.py` - Display bar counts in output
  - `src/edm/evaluation/evaluators/structure.py` - Support bar-based evaluation metrics
  - New module: `src/edm/analysis/bars.py` - Bar/measure utilities (extensible to beat grid)
