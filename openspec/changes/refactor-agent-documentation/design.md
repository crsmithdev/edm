## Context
CLAUDE.md currently contains extensive Python-specific coding standards (lines 74-135) that duplicate nearly all content in docs/python-style.md. This creates maintenance burden (changes must be made in two places) and violates single source of truth principles. Additionally, mixing language-agnostic workflow guidance with Python-specific syntax makes CLAUDE.md harder to navigate and maintain.

This proposal splits Python guidelines cleanly between the two files.

## Goals / Non-Goals

**Goals:**
- Establish single source of truth for Python guidelines
- Separate language-agnostic from Python-specific guidance
- Reduce CLAUDE.md length while preserving all information
- Add code examples to docs/python-style.md for every guideline

**Non-Goals:**
- Change the actual coding standards (keep existing rules)
- Create new documentation files (work with existing structure)
- Audit other documentation files (only CLAUDE.md and python-style.md)
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

### Decision: Add inline navigation in CLAUDE.md
Add reference to docs/python-style.md where Python details are removed from CLAUDE.md.

**Example:**
```markdown
# Detailed Guidelines

- **Python style**: See `docs/python-style.md`
- **Testing**: See `docs/testing.md`
- **Project structure**: See `docs/project-structure.md`
```

**Rationale:**
- Clear signposting to detailed Python guide
- Maintains high-level tool selection in CLAUDE.md
- Follows existing pattern in CLAUDE.md

### Decision: Preserve all content, just reorganize
Don't delete guidelines, only move and enhance them. Be conservative - when in doubt, keep content.

**Migration checklist:**
- [ ] Every line in current CLAUDE.md Python sections accounted for
- [ ] All guidelines preserved in docs/python-style.md
- [ ] All guidelines enhanced with examples
- [ ] Navigation links added

**Rationale:**
- Prevents accidental loss of important guidelines
- Ensures agents still have access to all information
- Low-risk refactoring
- Better to keep slightly verbose than lose useful information

### Decision: Interactive execution with user approval
Execute changes interactively, presenting each change for user approval before applying.

**Process:**
1. Analyze section to be changed
2. Present current vs proposed change
3. Wait for user approval/modification
4. Apply approved change
5. Move to next section

**Rationale:**
- User maintains control over documentation changes
- Prevents unintended removal of important content
- Allows for adjustments based on user feedback
- More conservative approach for documentation refactoring

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

**Risk:** Agents may not follow link to python-style.md and miss details
- **Mitigation:** Link clearly labeled in CLAUDE.md "Detailed Guidelines" section
- **Mitigation:** Agents already efficient at reading multiple files
- **Mitigation:** High-level tool selection still in CLAUDE.md

**Risk:** Losing content during migration
- **Mitigation:** Explicit checklist in tasks.md
- **Mitigation:** Interactive execution with user approval for each section
- **Mitigation:** Validation step verifies all content preserved
- **Mitigation:** Conservative approach - when in doubt, keep content
- **Mitigation:** Git history preserves old version

## Migration Plan

### Phase 1: Audit
1. Document Python content duplication between CLAUDE.md and docs/python-style.md
2. Create migration checklist

### Phase 2: Expand docs/python-style.md
3. Add code examples for all guidelines
4. Add detailed explanations
5. Ensure all CLAUDE.md Python content is covered

### Phase 3: Refactor CLAUDE.md (Interactive)
6. Present each section change for user approval
7. Replace detailed Python content with links to python-style.md
8. Apply approved changes

### Phase 4: Validation
9. Verify all content preserved
10. Verify single source of truth for Python guidelines
11. Verify navigation links work

## Open Questions
None - clear separation of concerns and straightforward refactoring.
