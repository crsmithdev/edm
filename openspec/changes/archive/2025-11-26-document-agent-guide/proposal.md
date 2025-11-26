# Change: Structured Documentation for Agents and Humans

## Why
Documentation is currently fragmented across README.md, CLAUDE.md, docs/, and openspec files, making it difficult for both AI agents and humans to find information efficiently. Agents must search multiple sources to understand architecture and workflows. Humans need quick reference without wading through explanatory content they already understand.

## What Changes
- Create `docs/agent-guide.md` as a navigation/index for AI agents (no content duplication, just smart routing to existing docs with context)
- Create `docs/architecture.md` for comprehensive system design, module organization, and design rationale
- Create `docs/cli-reference.md` for complete CLI command documentation with all options and examples
- Create `docs/development.md` for development workflows (testing, logging, code quality, evaluation framework)
- Create `TODO.md` in repo root for small tasks and quick fixes (< 1 hour)
- Create `IDEAS.md` in repo root for improvement ideas from reviews and potential future proposals
- Streamline `README.md` to essentials: brief description, system requirements, installation, quick start, basic commands (link to cli-reference.md for details)
- Link from CLAUDE.md to agent-guide.md as primary navigation point for agents
- Update CLAUDE.md with task management guidance (OpenSpec for features, TODO.md for small tasks, IDEAS.md for improvements)
- Establish single source of truth for each topic to prevent documentation drift

## Impact
- Affected specs: `documentation` (new capability)
- Affected code: None (documentation only)
- Affected files:
  - New: `docs/agent-guide.md`, `docs/architecture.md`, `docs/cli-reference.md`, `docs/development.md`, `TODO.md`, `IDEAS.md`
  - Modified: `README.md` (streamlined), `CLAUDE.md` (add links and task management guidance)
