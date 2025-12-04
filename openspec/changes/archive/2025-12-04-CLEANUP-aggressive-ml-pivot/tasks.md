# [CLEANUP] Implementation Tasks

## Phase 1: Immediate Cleanup

- [ ] 1.1 Remove empty stub modules
  - [ ] 1.1.1 Delete `src/edm/models/__init__.py`
  - [ ] 1.1.2 Delete `src/edm/features/__init__.py`

- [ ] 1.2 Remove unused exception
  - [ ] 1.2.1 Delete `ModelNotFoundError` from `src/edm/exceptions.py` (lines 22-25)

- [ ] 1.3 Remove temporary files
  - [ ] 1.3.1 Delete `temp.md`

- [ ] 1.4 Remove obsolete documentation
  - [ ] 1.4.1 Delete `docs/cli-patterns.md`
  - [ ] 1.4.2 Delete `docs/analysis-algorithms.md`
  - [ ] 1.4.3 Delete `docs/related-projects.md`

- [ ] 1.5 Archive algorithmic proposals
  - [ ] 1.5.1 Create `openspec/archive/2025-12-04-algorithmic-proposals/` directory
  - [ ] 1.5.2 Move `BEATSYNC-beat-synchronized-energy/` to archive
  - [ ] 1.5.3 Move `ECLUSTER-energy-envelope-clustering/` to archive
  - [ ] 1.5.4 Move `ENERGY-multi-feature-energy-detection/` to archive
  - [ ] 1.5.5 Move `MULTIENG-multi-feature-energy/` to archive
  - [ ] 1.5.6 Move `REFINE-energy-boundary-refinement/` to archive
  - [ ] 1.5.7 Move `TEMPORAL-energy-context/` to archive
  - [ ] 1.5.8 Move `SEGACC-segmentation-accuracy/` to archive
  - [ ] 1.5.9 Create archive README explaining why these were archived

- [ ] 1.6 Clean generated data
  - [ ] 1.6.1 Remove `data/generated/*.yaml` (6 files, not tracked)

## Phase 2: CLI Cleanup

- [ ] 2.1 Remove unused CLI flags from analyze command
  - [ ] 2.1.1 Remove `--offline` flag from `src/cli/commands/analyze.py`
  - [ ] 2.1.2 Remove `--annotations` flag from `src/cli/commands/analyze.py`
  - [ ] 2.1.3 Update tests that reference these flags

- [ ] 2.2 Remove trivial slash command
  - [ ] 2.2.1 Delete `.claude/commands/deps.md`

- [ ] 2.3 Update specs for CLI changes
  - [ ] 2.3.1 Update `openspec/specs/cli/spec.md` to reflect removed flags

## Phase 3: Test Cleanup

- [ ] 3.1 Evaluate JAMS test fixtures
  - [ ] 3.1.1 Check if `tests/estimations/beat_*.jams` are still used
  - [ ] 3.1.2 Remove JAMS fixtures if orphaned (5 files)

## Phase 4: Verification

- [ ] 4.1 Run full test suite
  - [ ] 4.1.1 `uv run pytest` - ensure all tests pass

- [ ] 4.2 Run type checking
  - [ ] 4.2.1 `uv run mypy src/` - ensure no errors from removed code

- [ ] 4.3 Update documentation index
  - [ ] 4.3.1 Update `docs/` index if one exists

## Phase 5: Git Commit

- [ ] 5.1 Stage all changes
- [ ] 5.2 Commit with message: "cleanup: aggressive ml-pivot cleanup"
- [ ] 5.3 Update CLEANUP proposal status to "deployed"
