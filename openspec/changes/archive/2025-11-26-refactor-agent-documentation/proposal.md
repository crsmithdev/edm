# Change: Split Python Guidelines Between CLAUDE.md and docs/python-style.md

## Why
CLAUDE.md and docs/python-style.md have ~95% duplication of Python-specific coding standards (lines 74-135 in CLAUDE.md duplicate nearly all of python-style.md). This violates single source of truth and creates maintenance burden. Additionally, CLAUDE.md mixes language-agnostic guidelines (git workflow, quality checkpoints) with Python-specific syntax rules, making it harder to maintain.

The agent documentation should follow a clear separation: CLAUDE.md for language-agnostic workflow guidance and high-level tool selection, with docs/python-style.md for Python-specific syntax, configuration, and code examples.

## What Changes
- Refactor CLAUDE.md to keep only language-agnostic guidelines and high-level tool selection
- Move Python-specific syntax, configuration, and detailed rules from CLAUDE.md to docs/python-style.md
- Expand docs/python-style.md with code examples for every guideline (both good and bad examples)
- Add navigation links in CLAUDE.md pointing to docs/python-style.md for Python-specific details
- Preserve all existing guidelines, just reorganize and enhance with examples

## Impact
- Affected specs: `documentation` (refinement)
- Affected code: None (documentation only)
- Affected files:
  - Modified: `CLAUDE.md` (remove Python-specific details, add navigation to python-style.md)
  - Modified: `docs/python-style.md` (expand with examples and details from CLAUDE.md)
