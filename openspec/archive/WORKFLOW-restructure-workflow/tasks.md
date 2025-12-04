# Tasks: Workflow Restructure

## 1. Pre-Migration Investigation

- [x] 1.1 Run `uv run edm analyze --help` to verify `--annotations` flag exists
- [x] 1.2 Analyze sample file, inspect YAML output for raw data structure
- [x] 1.3 Check if raw data includes per-detector breakdown (not implemented, deferred)
- [x] 1.4 Audit existing slash commands for `.claude/contexts/annotate.xml` references
- [x] 1.5 Document current workflow

## 2. CLI Enhancement (if needed)

- [x] 2.1 Skipped - existing raw data structure sufficient

## 3. Shell Script Implementation

- [x] 3.1 Create `.claude/scripts/analyze.sh` with file analysis loop
- [x] 3.2 Implement `sanitize_name()` function for filename normalization
- [x] 3.3 Create `.claude/scripts/evaluate.sh` with directory arg handling
- [x] 3.4 Simplify `.claude/scripts/annotate.sh`: remove subcommands, merge-only logic
- [x] 3.5 Add merge safety (preserve user edits, update raw data)
- [x] 3.6 Make all scripts executable: `chmod +x .claude/scripts/*.sh`

## 4. Context Consolidation

- [x] 4.1 Create `.claude/contexts/evaluation.xml` with merged content
- [x] 4.2 Include CLI usage, directory layout, evaluation metrics, analysis cascade
- [x] 4.3 Add safety rules (never delete annotations, merge strategy)
- [x] 4.4 Document output format with raw sections
- [x] 4.5 Validate XML syntax

## 5. Slash Command Updates

- [x] 5.1 Create `.claude/commands/analyze.md` invoking `analyze.sh`
- [x] 5.2 Update `.claude/commands/annotate.md`: remove subcommands, reference evaluation context
- [x] 5.3 Update `.claude/commands/evaluate.md` invoking `evaluate.sh`
- [x] 5.4 Test commands: `/analyze`, `/annotate`, `/evaluate`

## 6. Context Cleanup

- [x] 6.1 Delete `.claude/contexts/annotate.xml`
- [x] 6.2 Delete `.claude/contexts/evaluate.xml`
- [x] 6.3 Update `.gitignore` to exclude `data/generated/`

## 7. Testing

- [x] 7.1 Test `/analyze ~/music/test.flac` → verified `data/generated/test.yaml` created
- [x] 7.2 Test `/annotate` on new file → verified created in `data/annotations/`
- [x] 7.3 Edit annotation structure, re-run `/analyze`, then `/annotate` → verified merge preserves edits
- [x] 7.4 Test `/evaluate` → verified metrics output
- [x] 7.5 Skipped - evaluate works with directories only (current implementation)
- [x] 7.6 Tested graceful handling of empty directories

## 8. Documentation

- [ ] 8.1 Update README.md with new workflow commands (deferred)
- [ ] 8.2 Document output format in evaluation context (completed via context file)
- [ ] 8.3 Troubleshooting covered in command descriptions

## 9. Spec Updates

- [x] 9.1 Update `openspec/specs/cli/spec.md`: add evaluate and annotations requirements
- [x] 9.2 Update `openspec/specs/analysis/spec.md`: document raw data format
- [x] 9.3 Validate specs against implemented behavior
