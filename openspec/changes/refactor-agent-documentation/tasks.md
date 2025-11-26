## 1. Documentation Audit
- [ ] 1.1 Identify all duplicated content between CLAUDE.md and docs/python-style.md
- [ ] 1.2 Identify duplication across CLAUDE.md, docs/*.md, and openspec/project.md
- [ ] 1.3 Identify verbose or unnecessary content in agent documentation (be conservative)
- [ ] 1.4 Document findings and create consolidation plan

## 2. CLAUDE.md Refactoring (Interactive with User Approval)
- [ ] 2.1 Present current "Code Style" section vs proposed change, get approval
- [ ] 2.2 Present current "Type Hints" section vs proposed change, get approval
- [ ] 2.3 Present current "Testing" section vs proposed change, get approval
- [ ] 2.4 Present current "Error Handling" section vs proposed change, get approval
- [ ] 2.5 Present current "Async" section vs proposed change, get approval
- [ ] 2.6 Present current "Data Validation" section vs proposed change, get approval
- [ ] 2.7 Present current "Logging" section vs proposed change, get approval
- [ ] 2.8 Present current "Documentation" section vs proposed change, get approval
- [ ] 2.9 Present navigation section addition, get approval
- [ ] 2.10 Apply approved changes to CLAUDE.md

## 3. docs/python-style.md Expansion
- [ ] 3.1 Add code examples for modern type hints
- [ ] 3.2 Add code examples for import organization
- [ ] 3.3 Add code examples for error handling patterns
- [ ] 3.4 Add code examples for async patterns
- [ ] 3.5 Add code examples for Pydantic validation
- [ ] 3.6 Add code examples for logging patterns
- [ ] 3.7 Add configuration examples (Ruff, mypy)
- [ ] 3.8 Ensure all content from CLAUDE.md is preserved with better examples

## 4. Cross-Document Deduplication (Interactive with User Approval)
- [ ] 4.1 Identify duplication between docs/python-style.md and docs/testing.md
- [ ] 4.2 Present proposed deduplication for docs/testing.md, get approval
- [ ] 4.3 Identify duplication between docs/python-style.md and docs/project-structure.md
- [ ] 4.4 Present proposed deduplication for docs/project-structure.md, get approval
- [ ] 4.5 Identify duplication between CLAUDE.md and openspec/project.md
- [ ] 4.6 Present proposed deduplication for openspec/project.md, get approval
- [ ] 4.7 Apply approved deduplication changes

## 5. Validation
- [ ] 5.1 Verify CLAUDE.md contains only language-agnostic and high-level guidance
- [ ] 5.2 Verify docs/python-style.md has all Python-specific details with examples
- [ ] 5.3 Verify no content is lost during refactoring
- [ ] 5.4 Verify navigation links work and are complete
- [ ] 5.5 Verify single source of truth for each guideline
- [ ] 5.6 Test that agents can still find necessary information efficiently
