## MODIFIED Requirements

### Requirement: Agent Guide Documentation
The project SHALL provide comprehensive documentation in `docs/agent-guide.md` that serves as a unified reference for AI agents working with the codebase, with CLAUDE.md focusing on language-agnostic workflow guidance and linking to language-specific guides.

#### Scenario: Agent discovers high-level guidelines
- **WHEN** an agent reads CLAUDE.md
- **THEN** CLAUDE.md provides language-agnostic workflow guidance (git, quality checkpoints, agent patterns, task management) and tool selection with links to detailed guides

#### Scenario: Agent needs Python-specific guidance
- **WHEN** an agent needs Python syntax or code examples
- **THEN** CLAUDE.md links to docs/python-style.md which provides detailed syntax rules, configuration, and code examples

#### Scenario: Agent needs testing guidance
- **WHEN** an agent needs testing patterns or pytest specifics
- **THEN** CLAUDE.md links to docs/testing.md which provides detailed testing patterns

#### Scenario: Agent needs project structure guidance
- **WHEN** an agent needs directory layout or packaging information
- **THEN** CLAUDE.md links to docs/project-structure.md which provides detailed project organization

### Requirement: Single Source of Truth
The project documentation SHALL maintain exactly one canonical location for each guideline to prevent duplication and drift.

#### Scenario: Python syntax guidelines
- **WHEN** Python syntax guidelines are needed (type hints, imports, line length)
- **THEN** the canonical source is docs/python-style.md, not CLAUDE.md

#### Scenario: Language-agnostic workflow
- **WHEN** language-agnostic workflow guidance is needed (git, quality gates)
- **THEN** the canonical source is CLAUDE.md, not language-specific guides

#### Scenario: Tool selection vs usage
- **WHEN** high-level tool selection is needed (which tools to use)
- **THEN** CLAUDE.md provides tool selection with links to usage details

#### Scenario: Duplication audit
- **WHEN** documentation is updated
- **THEN** no guideline exists in multiple locations without clear primary/secondary relationship

### Requirement: Python Style Guide Enhancement
The file docs/python-style.md SHALL provide comprehensive Python-specific coding standards with code examples.

#### Scenario: Type hint examples
- **WHEN** an agent needs to write type hints
- **THEN** docs/python-style.md provides both good and bad code examples showing modern syntax vs deprecated syntax

#### Scenario: Import organization examples
- **WHEN** an agent needs to organize imports
- **THEN** docs/python-style.md provides complete code examples showing stdlib, third-party, and local imports

#### Scenario: Error handling examples
- **WHEN** an agent needs to implement error handling
- **THEN** docs/python-style.md provides code examples of custom exceptions and raise-from patterns

#### Scenario: Async pattern examples
- **WHEN** an agent needs to use async patterns
- **THEN** docs/python-style.md provides code examples using httpx and asyncio.TaskGroup

#### Scenario: Pydantic validation examples
- **WHEN** an agent needs to implement data validation
- **THEN** docs/python-style.md provides code examples using model_validator and Field()

#### Scenario: Logging pattern examples
- **WHEN** an agent needs to implement logging
- **THEN** docs/python-style.md provides code examples showing logger setup, lazy evaluation, and natural language messages

### Requirement: Documentation Navigation
CLAUDE.md SHALL provide clear navigation to detailed documentation guides.

#### Scenario: Navigation section at top
- **WHEN** an agent opens CLAUDE.md
- **THEN** a navigation section near the top lists all detailed guides with descriptions

#### Scenario: Inline references to details
- **WHEN** CLAUDE.md mentions a tool (Ruff, mypy, pytest)
- **THEN** it includes reference to detailed guide (e.g., "see python-style.md for configuration")

#### Scenario: Tool selection preserved
- **WHEN** an agent needs to know which tools to use
- **THEN** CLAUDE.md provides high-level tool selection without detailed configuration

### Requirement: Documentation Conciseness
CLAUDE.md SHALL be concise, focusing on workflow and high-level guidance without verbose explanations.

#### Scenario: Length reduction
- **WHEN** CLAUDE.md is refactored
- **THEN** length is reduced from ~250 lines to ~180 lines by moving Python-specific details to docs/python-style.md

#### Scenario: Actionable over philosophical
- **WHEN** guidelines are written
- **THEN** content is actionable (do this, use that) not philosophical (why this is good)

#### Scenario: Examples in specific guides
- **WHEN** code examples are needed
- **THEN** examples are in language-specific guides (docs/python-style.md), not CLAUDE.md
