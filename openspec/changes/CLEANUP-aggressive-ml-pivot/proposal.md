---
status: draft
created: 2025-12-04
---

# [CLEANUP] Aggressive ML-Pivot Cleanup

## Why

### Technical Debt from Algorithmic Approach
The codebase accumulated significant technical debt during the algorithmic exploration phase:
- 8 OpenSpec proposals for algorithmic energy detection improvements (superseded by ML)
- Empty stub modules awaiting implementation (`models/`, `features/`)
- Documentation for profiling features that have been removed
- Obsolete guides describing algorithmic approaches being replaced
- Unused CLI flags and temporary files

### Prepare for ML-First Implementation
The MLPIVOT proposal requires clean slate:
- `src/edm/models/` and `src/edm/features/` will be completely rewritten for ML
- Algorithmic energy detection proposals are obsolete (ML learns these patterns)
- Need clear codebase for new contributors to understand ML-first direction

### Reduce Maintenance Burden
- Fewer files to maintain during rapid ML development
- Less confusion about which approach is current
- Cleaner git history and smaller cognitive load

## What

### Files to Remove

**Empty stub modules:**
- `src/edm/models/__init__.py` - 5 lines, placeholder
- `src/edm/features/__init__.py` - 5 lines, placeholder

**Obsolete documentation:**
- `docs/cli-patterns.md` - Documents removed profiling command
- `docs/analysis-algorithms.md` - Describes algorithmic approaches being replaced
- `docs/related-projects.md` - Research doc, no longer relevant
- `temp.md` - Temporary cleanup notes

**Unused code:**
- `src/edm/exceptions.py` lines 22-25 - `ModelNotFoundError` exception (never used)
- `src/edm/config.py` line 76 - TODO comment about unimplemented TOML loading

**Trivial slash command:**
- `.claude/commands/deps.md` - Single command to run `uv sync --reinstall` (too trivial)

**Generated data (not tracked):**
- `data/generated/*.yaml` - Auto-generated, can be recreated

### OpenSpec Proposals to Archive

**Archive these 8 algorithmic proposals** (move to `openspec/archive/2025-12-04-algorithmic-proposals/`):
- `BEATSYNC-beat-synchronized-energy/` - Beat-synchronized energy analysis
- `ECLUSTER-energy-envelope-clustering/` - Energy envelope clustering
- `ENERGY-multi-feature-energy-detection/` - Multi-feature energy detection
- `MULTIENG-multi-feature-energy/` - Multi-feature energy
- `REFINE-energy-boundary-refinement/` - Energy boundary refinement
- `TEMPORAL-energy-context/` - Temporal energy context
- `SEGACC-segmentation-accuracy/` - Segmentation accuracy
- `HYBRID-msaf-energy-labeling/` - MSAF energy labeling (if exists)

**Reason:** All propose algorithmic improvements to energy detection. ML models will learn these patterns instead of hand-coding them.

**Keep active:**
- `MLPIVOT-learning-based-pivot/` - The new direction
- `XVAL-add-cross-validation/` - Relevant for ML evaluation
- `CLEANUP-aggressive-ml-pivot/` - This proposal

### CLI Changes (Breaking)

**Remove unused flags from `analyze` command:**
- `--offline` - Unused in current code
- `--annotations` - Redundant with `--format yaml` + output path
- Consider: `--no-color` - Replace with standard approach

**Files to modify:**
- `src/cli/commands/analyze.py` - Remove flag definitions and logic
- Tests using these flags

### Test Fixtures to Evaluate

**Check if still used, remove if orphaned:**
- `tests/estimations/beat_*.jams` (5 files) - JAMS format beat fixtures
  - Only remove if profiling tests were the sole user

### Dependencies to Review (Future)

**Evaluate during ML implementation:**
- `msaf>=0.1.80` - May keep as fallback or remove
- `librosa>=0.10.0` - May keep for feature extraction or remove
- `beat-this` - Keep, used for beat detection

**Not removing yet:** Need to evaluate during Phase 1-2 implementation.

## Impact

### Breaking Changes
**CLI:**
- `--offline` flag removed from `analyze` command (unused, low impact)
- `--annotations` flag removed from `analyze` command (low impact, alternative exists)

**Python API:**
- `ModelNotFoundError` exception removed (never used, zero impact)

### No Migration Needed
- Empty stub modules had no functionality to migrate
- Documentation removal doesn't affect code
- OpenSpec proposals archived, not deleted (can reference if needed)

### Risks

**Low risk:**
- Empty stubs have no dependents
- Unused flags have no users
- Algorithmic proposals are superseded, not wrong

**Medium risk:**
- Removing JAMS fixtures: Must verify no active tests depend on them
- Archiving 8 proposals: Ensure no external references (e.g., in commits, issues)

**Mitigation:**
- Run full test suite after cleanup
- Archive proposals instead of deleting (can recover if needed)
- Document archived proposals in archive README

### Affected Specs

- `openspec/specs/cli/spec.md` - Update to reflect removed flags
- `openspec/specs/development-workflow/spec.md` - Remove references to algorithmic proposals

## Benefits

1. **Clarity:** New contributors see ML-first direction immediately
2. **Reduced cognitive load:** 8 fewer proposals to understand
3. **Clean slate:** `models/` and `features/` ready for ML implementation
4. **Less maintenance:** Fewer files to keep updated
5. **Faster CI:** Fewer test files if JAMS fixtures removed

## Drawbacks

1. **Lost historical context:** Algorithmic proposals archived (mitigated by keeping in archive/)
2. **Potential regret:** If ML fails, algorithmic improvements unavailable (low probability)
3. **Documentation gaps:** Some users may rely on removed docs (mitigated by keeping essential docs)

## Implementation Order

1. **Phase 1 (Immediate):**
   - Remove empty stub modules
   - Remove `ModelNotFoundError`
   - Delete `temp.md`
   - Remove obsolete documentation
   - Archive 8 algorithmic proposals

2. **Phase 2 (Before MLPIVOT Phase 1):**
   - Remove unused CLI flags
   - Verify and remove JAMS fixtures if orphaned
   - Remove trivial `/deps` command

3. **Phase 3 (During MLPIVOT implementation):**
   - Evaluate and remove algorithmic dependencies (msaf, librosa fallbacks)
