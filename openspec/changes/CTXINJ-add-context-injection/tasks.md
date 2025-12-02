# Tasks: Context Injection & Documentation

## 1. Context Injection Hook

- [x] 1.1 Rewrite `.claude/hooks/inject-edm-context.sh` with category detection (stdin-based)
- [x] 1.2 Create `.claude/contexts/` directory
- [x] 1.3 Create `openspec.xml` context file (~80 lines: workflow, spec format, validation)
- [x] 1.4 Create `audio.xml` context file (~100 lines: algorithms, fallbacks, types, layout)
- [x] 1.5 Create `python.xml` context file (~60 lines: code style, Typer/CLI, testing)
- [x] 1.6 Test hook with sample prompts from each category

## 2. High-Priority Documentation

- [x] 2.1 Create `docs/analysis-algorithms.md`
  - Fallback chains (beat_this → librosa, MSAF → energy)
  - Energy thresholds for section labeling
  - BeatGrid anchor model explanation
- [x] 2.2 Expand `docs/architecture.md`
  - Module dependency diagram
  - Two-tier abstraction pattern (detector vs analyze_*)
  - Audio caching strategy (io/audio.py)

## 3. Medium-Priority Documentation

- [x] 3.1 Create `docs/cli-patterns.md`
  - Typer conventions specific to this project
  - Command registration pattern
  - Rich output formatting (tables, JSON, YAML)
- [x] 3.2 Create `src/edm/analysis/README.md`
  - Module-level overview
  - Data flow between components
  - Key types and relationships
