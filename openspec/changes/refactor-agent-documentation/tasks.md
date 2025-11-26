## 1. Documentation Audit
- [ ] 1.1 Identify all duplicated content between CLAUDE.md and docs/python-style.md
- [ ] 1.2 Identify duplication across CLAUDE.md, docs/*.md, and openspec/project.md
- [ ] 1.3 Identify verbose or unnecessary content in agent documentation
- [ ] 1.4 Document findings and create consolidation plan

## 2. CLAUDE.md Refactoring
- [ ] 2.1 Remove Python-specific syntax details (line length, quote style, import order specifics)
- [ ] 2.2 Remove Python-specific type hint syntax (`list[str]` vs `List[str]` examples)
- [ ] 2.3 Remove Python-specific testing details (pytest specifics, naming conventions)
- [ ] 2.4 Remove Python-specific error handling patterns
- [ ] 2.5 Remove Python-specific async/validation/logging details
- [ ] 2.6 Keep high-level tool selection (Ruff, mypy, pytest, Pydantic, httpx)
- [ ] 2.7 Add navigation section with links to detailed guides
- [ ] 2.8 Reduce CLAUDE.md from ~250 lines to ~180 lines

## 3. docs/python-style.md Expansion
- [ ] 3.1 Add code examples for modern type hints
- [ ] 3.2 Add code examples for import organization
- [ ] 3.3 Add code examples for error handling patterns
- [ ] 3.4 Add code examples for async patterns
- [ ] 3.5 Add code examples for Pydantic validation
- [ ] 3.6 Add code examples for logging patterns
- [ ] 3.7 Add configuration examples (Ruff, mypy)
- [ ] 3.8 Ensure all content from CLAUDE.md is preserved with better examples

## 4. Cross-Document Deduplication
- [ ] 4.1 Remove duplication between docs/python-style.md and docs/testing.md (if any)
- [ ] 4.2 Remove duplication between docs/python-style.md and docs/project-structure.md (if any)
- [ ] 4.3 Remove duplication between CLAUDE.md and openspec/project.md (if any)
- [ ] 4.4 Consolidate overlapping content, keeping in most appropriate location

## 5. Validation
- [ ] 5.1 Verify CLAUDE.md contains only language-agnostic and high-level guidance
- [ ] 5.2 Verify docs/python-style.md has all Python-specific details with examples
- [ ] 5.3 Verify no content is lost during refactoring
- [ ] 5.4 Verify navigation links work and are complete
- [ ] 5.5 Verify single source of truth for each guideline
- [ ] 5.6 Test that agents can still find necessary information efficiently
