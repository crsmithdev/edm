# Change: Add Context-Aware Prompt Injection

## Why

Current static XML context injection doesn't adapt to task type. OpenSpec tasks need workflow details, audio tasks need algorithm info, general Python tasks need style conventions. Injecting everything wastes tokens; injecting too little loses precision.

## What Changes

- Add category-detecting context injection hook (3 categories: openspec, audio, python)
- Create specialized context files with task-relevant information
- Add supporting documentation for authoritative references (analysis algorithms, architecture, CLI patterns)

## Impact

- Affected specs: development-workflow
- Affected code:
  - `.claude/hooks/inject-edm-context.sh` - rewrite with detection logic
  - `.claude/contexts/` - new directory with 3 XML context files
  - `docs/analysis-algorithms.md` - new documentation
  - `docs/architecture.md` - expand existing
  - `docs/cli-patterns.md` - new documentation
  - `src/edm/analysis/README.md` - new module overview
