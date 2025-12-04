# [XVAL] Add Cross-Validation Framework

status: draft

## Why

When structure boundaries don't align with whole-bar positions (e.g., a drop at bar 8.34), it signals an error in either BPM detection or structure detection. Currently these analyses run independently with no cross-validation. By detecting misalignment patterns, we can:

1. Diagnose error sources (BPM vs structure vs phase vs downbeat)
2. Auto-correct high-confidence errors
3. Flag ambiguous cases for review
4. Improve overall analysis accuracy

Research from [ISMIR/TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.167) shows downbeat alignment improved structure accuracy from 27% to 43%.

## What Changes

- Add pluggable validator framework (`src/edm/analysis/validation/`)
- Implement beat/structure alignment validator as first validator
- Implement downbeat/structure alignment validator using `BeatGrid` (separate from BPM)
- Add `--validate` flag to CLI analyze command
- Report alignment metrics and error patterns in analysis output
- Cross-validate four independent signals:
  - `ComputedBPM` - tempo
  - `BeatGrid` - beat positions (index-based)
  - `BeatGrid.first_downbeat` - downbeat phase
  - Structure sections - detected sections
- Initially flag-only mode; auto-correction enabled once validated

## Impact

- Affected specs: `analysis` (adds cross-validation requirements)
- Affected code:
  - `src/edm/analysis/validation/` (new module)
  - `src/cli/commands/analyze.py` (integration)
  - `src/cli/main.py` (new CLI flags)
