# Change: Refactor Agent Documentation Standards

## Why
CLAUDE.md and docs/python-style.md have ~95% duplication of Python-specific coding standards (lines 74-135 in CLAUDE.md duplicate nearly all of python-style.md). This violates single source of truth and creates maintenance burden. Additionally, CLAUDE.md mixes language-agnostic guidelines (git workflow, quality checkpoints) with Python-specific syntax rules, making it harder to maintain and extend to other languages if needed.

The agent documentation should follow a clear hierarchy: CLAUDE.md for high-level tool selection and workflows, with links to language-specific guides for syntax and examples.

## What Changes
- Refactor CLAUDE.md to keep only language-agnostic guidelines and high-level tool selection
- Move Python-specific syntax, examples, and detailed rules to docs/python-style.md
- Expand docs/python-style.md with code examples and detailed explanations
- Add clear navigation links from CLAUDE.md to docs/python-style.md, docs/testing.md, docs/project-structure.md
- Audit all agent documentation (CLAUDE.md, docs/*.md, openspec/project.md) for duplication and unnecessary verbosity
- Remove or consolidate duplicated content
- Ensure each piece of information has exactly one canonical location

## Impact
- Affected specs: `documentation` (refinement)
- Affected code: None (documentation only)
- Affected files:
  - Modified: `CLAUDE.md` (remove Python-specific details, add navigation links)
  - Modified: `docs/python-style.md` (expand with examples and details from CLAUDE.md)
  - Modified: `docs/testing.md` (if duplication found)
  - Modified: `docs/project-structure.md` (if duplication found)
  - Modified: `openspec/project.md` (if duplication found)
