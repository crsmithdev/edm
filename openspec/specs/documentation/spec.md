# documentation Specification

## Purpose
TBD - created by archiving change refactor-agent-documentation. Update Purpose after archive.
## Requirements
### Requirement: Python Guidelines Separation
The project SHALL separate Python-specific coding standards from language-agnostic workflow guidance, with CLAUDE.md containing workflow and high-level tool selection, and docs/python-style.md containing Python-specific syntax, configuration, and examples.

#### Scenario: Agent discovers high-level workflow guidelines
- **WHEN** an agent reads CLAUDE.md
- **THEN** CLAUDE.md provides language-agnostic workflow guidance (git, quality checkpoints, agent patterns, task management) without Python-specific syntax details

#### Scenario: Agent needs Python-specific guidance
- **WHEN** an agent needs Python syntax or code examples
- **THEN** CLAUDE.md links to docs/python-style.md which provides detailed syntax rules, configuration, and code examples

#### Scenario: Tool selection vs configuration
- **WHEN** an agent needs to know which tools to use
- **THEN** CLAUDE.md provides high-level tool selection (Ruff, mypy, pytest) with links to python-style.md for configuration details

### Requirement: Single Source of Truth for Python Guidelines
The project documentation SHALL maintain exactly one canonical location for each Python guideline to prevent duplication between CLAUDE.md and docs/python-style.md.

#### Scenario: Python syntax guidelines
- **WHEN** Python syntax guidelines are needed (type hints, imports, line length)
- **THEN** the canonical source is docs/python-style.md, not CLAUDE.md

#### Scenario: Language-agnostic workflow
- **WHEN** language-agnostic workflow guidance is needed (git, quality checkpoints)
- **THEN** the canonical source is CLAUDE.md, not python-style.md

#### Scenario: Python configuration details
- **WHEN** Python tool configuration is needed (Ruff settings, mypy options)
- **THEN** the canonical source is docs/python-style.md, not CLAUDE.md

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
CLAUDE.md SHALL provide clear navigation to docs/python-style.md for Python-specific details.

#### Scenario: Inline reference to python-style.md
- **WHEN** CLAUDE.md removes Python-specific content
- **THEN** it includes clear link to docs/python-style.md in "Detailed Guidelines" section

#### Scenario: Examples in python-style.md
- **WHEN** Python code examples are needed
- **THEN** examples are in docs/python-style.md, not CLAUDE.md

