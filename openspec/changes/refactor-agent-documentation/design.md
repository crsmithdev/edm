## Context
CLAUDE.md currently contains extensive Python-specific coding standards (lines 74-135) that duplicate nearly all content in docs/python-style.md. This creates maintenance burden (changes must be made in two places) and violates single source of truth principles. Additionally, mixing language-agnostic workflow guidance with Python-specific syntax makes CLAUDE.md harder to navigate and maintain.

This proposal establishes clear documentation boundaries and eliminates duplication across agent documentation.

## Goals / Non-Goals

**Goals:**
- Establish single source of truth for each guideline
- Separate language-agnostic from language-specific guidance
- Reduce CLAUDE.md length while preserving all information
- Add code examples to docs/python-style.md
- Eliminate duplication across all agent documentation
- Improve navigation with clear links

**Non-Goals:**
- Change the actual coding standards (keep existing rules)
- Create new documentation files (work with existing structure)
- Document undocumented areas (only refactor existing content)
- Change OpenSpec workflow (documentation refactoring only)

## Decisions

### Decision: Language-agnostic vs language-specific split
Keep language-agnostic guidelines in CLAUDE.md, move Python-specific details to docs/python-style.md.

**CLAUDE.md keeps:**
- Git workflow (branch naming, commits, push timing)
- Quality checkpoints (before commit, before proposal, before task complete)
- Agent orchestration (workflow patterns, model selection)
- Task management (OpenSpec vs TODO vs IDEAS)
- Error recovery playbooks (language-agnostic debugging process)
- Critical file groups (what files to read together)
- **Tool selection** (Ruff, mypy, pytest, Pydantic) - high level only

**docs/python-style.md gets:**
- Specific Ruff configuration and usage
- Exact line length, quote preferences, import organization
- Type hint syntax examples (`list[str]` vs `List[str]`)
- pytest naming conventions and fixture patterns
- Pydantic v2 specific features (`model_validator`, `Field()`)
- Async patterns (`asyncio.TaskGroup`, httpx usage)
- Logging patterns and examples
- Error handling code examples
- All code examples demonstrating guidelines

**Rationale:**
- CLAUDE.md becomes navigation hub for agents
- Python-specific details don't belong in language-agnostic workflow guide
- Future language support easier (just add docs/rust-style.md, etc.)
- Single source of truth per guideline
- CLAUDE.md stays under 200 lines (currently ~250)

### Decision: Expand docs/python-style.md with examples
Add code examples to python-style.md for every guideline.

**Before (current):**
```markdown
## Type Hints
- Use modern syntax: `list[str]` not `List[str]`
```

**After:**
```markdown
## Type Hints

Use modern Python 3.10+ type hint syntax throughout:

**Good:**
```python
def process(items: list[str], count: int | None = None) -> dict[str, int]:
    """Process items and return counts."""
    ...
```

**Bad:**
```python
from typing import List, Optional, Dict

def process(items: List[str], count: Optional[int] = None) -> Dict[str, int]:
    ...
```
```

**Rationale:**
- Examples clarify intent better than rules alone
- Agents can copy-paste correct patterns
- Shows both good and bad examples
- Reduces ambiguity

### Decision: Navigation section in CLAUDE.md
Add clear navigation to detailed guides at top of CLAUDE.md.

**Structure:**
```markdown
# Development Guidelines

For detailed coding standards and examples, see:
- **Python style**: `docs/python-style.md` - Syntax, tools, code examples
- **Testing patterns**: `docs/testing.md` - pytest, fixtures, coverage
- **Project structure**: `docs/project-structure.md` - Directory layout, packaging

## Tool Selection
- **Formatter/Linter**: Ruff (see python-style.md for configuration)
- **Type Checking**: mypy strict mode (see python-style.md for usage)
- **Testing**: pytest (see testing.md for patterns)
...
```

**Rationale:**
- Agents see navigation immediately
- Clear signposting to detailed guides
- Tool selection at high level, details elsewhere
- Follows same pattern as document-agent-guide proposal

### Decision: Audit all agent documentation for duplication
Check CLAUDE.md, docs/*.md, and openspec/project.md for overlapping content.

**Audit areas:**
- CLAUDE.md vs docs/python-style.md (known duplication)
- CLAUDE.md vs docs/testing.md (potential overlap on pytest)
- CLAUDE.md vs docs/project-structure.md (potential overlap on src/ layout)
- CLAUDE.md vs openspec/project.md (potential overlap on tech stack, conventions)
- docs/python-style.md vs docs/testing.md (potential overlap)

**Consolidation rules:**
1. Keep content in most specific location (python-style.md over CLAUDE.md)
2. If truly relevant to multiple audiences, keep in most general location with link from specific
3. Remove verbose explanations not useful to agents
4. Keep actionable guidelines, remove "why this is good" philosophy

**Rationale:**
- Prevents duplication from creeping back
- Establishes clear ownership per topic
- Reduces total documentation volume

### Decision: Preserve all content, just reorganize
Don't delete guidelines, only move and enhance them.

**Migration checklist:**
- [ ] Every line in current CLAUDE.md Python sections accounted for
- [ ] All guidelines preserved in docs/python-style.md
- [ ] All guidelines enhanced with examples
- [ ] Navigation links added

**Rationale:**
- Prevents accidental loss of important guidelines
- Ensures agents still have access to all information
- Low-risk refactoring

## Structure

### CLAUDE.md After Refactoring (~180 lines)

```markdown
<!-- OPENSPEC block -->

# Development Guidelines

For detailed coding standards:
- Python: docs/python-style.md
- Testing: docs/testing.md
- Project: docs/project-structure.md

# Task Execution
[Existing - enhanced model selection]

# Development Context
[Existing]

# Interaction Style
[Existing]

# Git Commits
[Existing - one-line style]

# Git Workflow
[Existing - branch naming, granularity, push timing]

# Tool Selection
- Formatter/Linter: Ruff (see python-style.md)
- Type Checking: mypy strict
- Testing: pytest (see testing.md)
- Validation: Pydantic v2
- Async: httpx, asyncio.TaskGroup
- Project: src/ layout (see project-structure.md)

# Quality Checkpoints
[Existing]

# Critical Files for Common Tasks
[Existing]

# Error Recovery Playbooks
[Existing]

# Agent Workflow Patterns
[Existing]

# Task Management
[Existing]

# Commands Reference
[Existing]
```

### docs/python-style.md After Expansion (~120 lines)

```markdown
# Python Style Guide

## Code Style

### Formatter and Linter
Use Ruff for both formatting and linting:

```bash
uv run ruff format .      # Format code
uv run ruff check --fix .  # Lint with auto-fix
```

### Line Length
88 characters (Black/Ruff default)

### Quotes
Double quotes preferred for strings

### Import Organization
stdlib → third-party → local, sorted alphabetically within groups

**Example:**
```python
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from edm.analysis import bpm
from edm.config import settings
```

## Type Hints

[Examples of modern syntax, TypeAlias, etc.]

## Error Handling

[Examples of custom exceptions, raise from, etc.]

## Async Patterns

[Examples of httpx, TaskGroup, etc.]

## Data Validation

[Examples of Pydantic v2, model_validator, Field(), etc.]

## Logging

[Examples of logger setup, lazy evaluation, natural language, etc.]

## Documentation

[Examples of Google-style docstrings]
```

## Risks / Trade-offs

**Risk:** Agents may not follow links and miss details
- **Mitigation:** Clear navigation at top of CLAUDE.md
- **Mitigation:** Agents already efficient at reading multiple files
- **Mitigation:** Critical info (tool selection) still in CLAUDE.md at high level

**Risk:** Breaking existing agent muscle memory
- **Mitigation:** Tool selection still in CLAUDE.md, just less detail
- **Mitigation:** Gradual adoption - old patterns still work
- **Mitigation:** Navigation makes new structure discoverable

**Trade-off:** More files to maintain vs cleaner separation
- **Decision:** Cleaner separation wins - already have the files
- **Pro:** Single source of truth reduces overall maintenance
- **Con:** Must keep links accurate
- **Mitigation:** Links are simple, rarely change

**Risk:** Losing content during migration
- **Mitigation:** Explicit checklist in tasks.md
- **Mitigation:** Validation step verifies all content preserved
- **Mitigation:** Git history preserves old version

## Migration Plan

### Phase 1: Audit
1. Document all duplicated content
2. Document unnecessary verbose content
3. Create migration checklist

### Phase 2: Expand docs/python-style.md
4. Add code examples for all guidelines
5. Add detailed explanations
6. Ensure all CLAUDE.md Python content is covered

### Phase 3: Refactor CLAUDE.md
7. Add navigation section at top
8. Replace detailed Python content with tool selection + links
9. Verify length reduction (target ~180 lines)

### Phase 4: Cross-Document Deduplication
10. Remove duplication across all docs
11. Consolidate overlapping content

### Phase 5: Validation
12. Verify all content preserved
13. Verify single source of truth
14. Test agent navigation

## Open Questions
None - clear separation of concerns and straightforward refactoring.
