## ADDED Requirements

### Requirement: Agent Navigation Guide
The project SHALL provide `docs/agent-guide.md` as a navigation/index for AI agents that routes to topic-specific documentation without duplicating content.

#### Scenario: Agent discovers documentation structure
- **WHEN** an agent reads agent-guide.md
- **THEN** the guide provides a "Quick Reference by Task" section linking to appropriate detailed documentation with brief context

#### Scenario: Agent finds relevant documentation
- **WHEN** an agent needs information on a specific topic (CLI, architecture, development)
- **THEN** agent-guide.md routes to the appropriate detailed doc (cli-reference.md, architecture.md, development.md) with context about what information is available

#### Scenario: Agent locates code by feature
- **WHEN** an agent needs to find implementation code
- **THEN** agent-guide.md provides file paths organized by feature (BPM detection, evaluation, external services)

#### Scenario: Navigation guide avoids duplication
- **WHEN** agent-guide.md is created or updated
- **THEN** content consists of navigation, links, and brief context only (no duplication of detailed content from other docs)

### Requirement: Architecture Documentation
The project SHALL provide `docs/architecture.md` with comprehensive system design, module organization, and design rationale.

#### Scenario: Understanding system design
- **WHEN** an agent or human needs to understand the system architecture
- **THEN** architecture.md provides overview, module organization, data flow, key design decisions, and tech stack choices with rationale

#### Scenario: Finding design rationale
- **WHEN** understanding why a design choice was made (e.g., why madmom for BPM, why cascading strategy)
- **THEN** architecture.md documents the rationale, trade-offs considered, and alternatives rejected

#### Scenario: Locating modules
- **WHEN** finding code for a specific capability
- **THEN** architecture.md documents module locations with file paths (e.g., `src/edm/analysis/bpm.py:45`)

### Requirement: CLI Reference Documentation
The project SHALL provide `docs/cli-reference.md` with complete command documentation for all CLI commands and options.

#### Scenario: Complete command reference
- **WHEN** an agent or human needs to use a CLI command
- **THEN** cli-reference.md provides complete documentation of command structure, all options, and usage examples for both analyze and evaluate subcommands

#### Scenario: Understanding command options
- **WHEN** determining which flags or options to use
- **THEN** cli-reference.md documents each option's purpose, valid values, and examples with expected output

#### Scenario: Configuration guidance
- **WHEN** configuring the CLI (environment variables, config files)
- **THEN** cli-reference.md documents configuration options and usage

### Requirement: Development Workflow Documentation
The project SHALL provide `docs/development.md` with comprehensive development workflows, testing, logging, and code quality information.

#### Scenario: Running tests
- **WHEN** an agent or human needs to run tests
- **THEN** development.md provides test commands, coverage requirements, and workflow guidance

#### Scenario: Understanding logging
- **WHEN** working with logging in the codebase
- **THEN** development.md documents logging patterns, configuration, and structlog usage

#### Scenario: Code quality tools
- **WHEN** running code quality checks
- **THEN** development.md documents tools (ruff, mypy, pytest), their configuration, and usage

#### Scenario: Evaluation framework
- **WHEN** running or understanding the evaluation system
- **THEN** development.md provides evaluation workflow, reference sources, and result interpretation

### Requirement: README Streamlining
The README.md SHALL provide quick reference information only, linking to detailed documentation for comprehensive information.

#### Scenario: Quick start for new users
- **WHEN** a user reads README.md
- **THEN** the file contains brief description (2-3 sentences), system requirements, installation steps, and quick start examples

#### Scenario: Linking to detailed docs
- **WHEN** README.md content is streamlined
- **THEN** links to detailed documentation (cli-reference.md, architecture.md, development.md) are provided for users needing more information

#### Scenario: Basic commands included
- **WHEN** a user needs to run basic commands
- **THEN** README.md includes most essential run/test commands with note to see cli-reference.md for complete documentation

#### Scenario: Explanatory content moved to topic docs
- **WHEN** README.md is reviewed for streamlining
- **THEN** explanatory content (tool rationale, design decisions) is moved to appropriate topic-specific documentation files

### Requirement: Single Source of Truth
The project documentation SHALL maintain a single source of truth for each topic to prevent documentation drift.

#### Scenario: No content duplication
- **WHEN** documentation is created or updated
- **THEN** detailed content exists in exactly one location (architecture.md for design, cli-reference.md for commands, etc.)

#### Scenario: Topic ownership
- **WHEN** a specific topic needs updating (CLI change, architecture change, workflow change)
- **THEN** updates occur in the single authoritative doc for that topic

#### Scenario: Cross-referencing instead of duplication
- **WHEN** multiple docs need to reference the same information
- **THEN** docs link to the authoritative source rather than duplicating content

### Requirement: Documentation Maintenance
The project SHALL keep documentation synchronized with code changes through the OpenSpec change management process.

#### Scenario: Documentation updates with code changes
- **WHEN** an OpenSpec change modifies CLI, architecture, or workflows
- **THEN** the change proposal includes tasks to update the appropriate documentation file (cli-reference.md, architecture.md, or development.md)

#### Scenario: Agent discovers documentation entry point
- **WHEN** an agent reads CLAUDE.md for project instructions
- **THEN** CLAUDE.md links to agent-guide.md as the primary navigation point for documentation

### Requirement: Documentation Format
All documentation SHALL use structured markdown with consistent formatting for both human and AI readability.

#### Scenario: Consistent markdown structure
- **WHEN** documentation is created or updated
- **THEN** content uses consistent heading hierarchy, code blocks with language tags, and tables for reference data

#### Scenario: File paths for navigation
- **WHEN** documentation references code
- **THEN** file paths use the format `path/to/file.py:line_number` to enable direct navigation

#### Scenario: Code examples with output
- **WHEN** documentation provides CLI examples or code samples
- **THEN** examples are in code blocks with appropriate language tags and include expected output or result descriptions

### Requirement: Task Tracking Files
The project SHALL provide `TODO.md` and `IDEAS.md` in the repo root for tracking small tasks and improvement ideas outside the OpenSpec workflow.

#### Scenario: Small tasks tracked in TODO.md
- **WHEN** a small task or quick fix (< 1 hour) is identified
- **THEN** it is added to TODO.md with checkboxes organized by priority (High/Medium/Low)

#### Scenario: Improvement ideas captured in IDEAS.md
- **WHEN** code reviews or architecture reviews generate improvement suggestions
- **THEN** ideas are captured in IDEAS.md with categorization (Ready for Proposal / Under Consideration / Icebox), effort estimation, and context

#### Scenario: Converting ideas to proposals
- **WHEN** an idea in IDEAS.md matures and warrants full implementation
- **THEN** an OpenSpec proposal is created and the idea is referenced or moved

#### Scenario: Task management workflow documented
- **WHEN** an agent reads CLAUDE.md for project conventions
- **THEN** task management guidance is provided: OpenSpec for features/breaking changes, TODO.md for small tasks, IDEAS.md for improvement ideas

#### Scenario: Agents use task tracking appropriately
- **WHEN** an agent identifies a small fix or improvement during work
- **THEN** the agent adds it to TODO.md (if small task) or IDEAS.md (if improvement idea) with appropriate context and categorization
