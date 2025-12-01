# Change: Review and Simplify Fallback Patterns

## Why
The codebase has fallback patterns that silently degrade quality when things break. We want errors to surface loudly so they can be fixed, not hidden behind fallbacks that produce inferior results.

## Current State
- **MSAF fallbacks**: Falls back to EnergyDetector on import or runtime errors
- **beat_this fallbacks**: Falls back to librosa (legitimate - GPU is optional)
- **matplotlib**: Conditional import for benchmarks (legitimate)
- **BPM workflow**: metadata → computed (not a fallback, just workflow)

## What Changes
- **Remove**: MSAF fallbacks entirely - make msaf required, fail loudly on errors
- **Keep**: beat_this → librosa fallback (GPU is genuinely optional)
- **Keep**: matplotlib conditional import (benchmark-only feature)
- **Keep**: CUDA → CPU auto-detection

## Impact
- MSAF issues surface immediately instead of silently degrading
- Simpler code paths in structure detection
- Clearer dependency requirements
- EnergyDetector may become dead code (evaluate removal)
