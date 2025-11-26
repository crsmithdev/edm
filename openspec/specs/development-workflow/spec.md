# development-workflow Specification

## Purpose
TBD - created by archiving change workflow-optimizations. Update Purpose after archive.
## Requirements
### Requirement: Development Workflow Slash Commands
The project SHALL provide slash commands for common development workflows to reduce manual command execution.

#### Scenario: Check all quality gates
- **WHEN** a developer or agent needs to verify code quality before committing
- **THEN** the `/check` command runs format, lint, type check, and tests in sequence and reports results

#### Scenario: Quick task capture
- **WHEN** a small task or bug is identified during work
- **THEN** the `/todo` command interactively prompts for priority and task description and adds to TODO.md

#### Scenario: Capture improvement idea
- **WHEN** a code review or analysis identifies a potential improvement
- **THEN** the `/idea` command interactively prompts for category and description and adds to IDEAS.md with context

#### Scenario: Project status overview
- **WHEN** checking overall project state
- **THEN** the `/status` command displays git status, active OpenSpec changes, TODO count, and IDEAS count

#### Scenario: Sync repository and dependencies
- **WHEN** starting work or after pulling changes
- **THEN** the `/sync` command pulls latest git changes and runs uv sync

#### Scenario: Clean build artifacts
- **WHEN** needing a fresh start or debugging environment issues
- **THEN** the `/clean` command removes __pycache__, .pytest_cache, .mypy_cache, .coverage, and htmlcov directories

### Requirement: Quality Checkpoints
The project SHALL define explicit quality checkpoints in CLAUDE.md that agents and developers must verify before critical actions.

#### Scenario: Pre-commit quality verification
- **WHEN** an agent is about to commit code
- **THEN** the agent verifies all tests pass, type checking passes, linting passes, no TODO comments without TODO.md entries, and relevant docs are updated

#### Scenario: Pre-proposal validation
- **WHEN** an agent is creating an OpenSpec proposal
- **THEN** the agent verifies existing specs are checked for duplicates, openspec/project.md is reviewed, `openspec list` shows no conflicts, and `openspec validate --strict` passes

#### Scenario: Pre-task-completion verification
- **WHEN** an agent is marking a task as complete
- **THEN** the agent verifies all tasks.md items are checked off, code examples in docs still work, and no commented-out code remains

### Requirement: Context Management Hints
The project SHALL document which files should be read together for common tasks to improve agent efficiency.

#### Scenario: BPM-related work context
- **WHEN** an agent is working on BPM detection features
- **THEN** CLAUDE.md indicates to read src/edm/analysis/bpm.py, src/edm/analysis/bpm_detector.py, and tests/test_analysis/test_bpm.py together

#### Scenario: External services context
- **WHEN** an agent is adding or modifying external service integrations
- **THEN** CLAUDE.md indicates to read all files in src/edm/external/, src/edm/config.py, and docs/architecture.md together

#### Scenario: Evaluation framework context
- **WHEN** an agent is working on evaluation features
- **THEN** CLAUDE.md indicates to read src/edm/evaluation/common.py, src/edm/evaluation/reference.py, and docs/cli-reference.md together

### Requirement: Error Recovery Playbooks
The project SHALL provide step-by-step recovery procedures in CLAUDE.md for common failure scenarios.

#### Scenario: Test failure recovery
- **WHEN** tests fail after code changes
- **THEN** CLAUDE.md provides steps: read full test output, identify root cause (logic vs test vs environment), fix root cause, re-run affected tests, re-run full suite if still failing

#### Scenario: Import error recovery
- **WHEN** import or dependency errors occur
- **THEN** CLAUDE.md provides steps: run uv sync, check for circular imports, verify virtual environment, check pyproject.toml for missing dependencies

#### Scenario: OpenSpec validation failure recovery
- **WHEN** openspec validate fails
- **THEN** CLAUDE.md provides steps: run with --strict flag, check scenario format (#### not ### or -), verify requirements have scenarios, use --json --deltas-only to debug

### Requirement: Agent Workflow Patterns
The project SHALL document standard patterns in CLAUDE.md for multi-agent orchestration and handoffs.

#### Scenario: Code review to implementation workflow
- **WHEN** requesting code review with implementation
- **THEN** CLAUDE.md documents pattern: spawn code-reviewer agent, capture findings, spawn implementation agent with context, track both in TodoWrite

#### Scenario: Analysis to proposal to implementation workflow
- **WHEN** requesting architecture changes
- **THEN** CLAUDE.md documents pattern: spawn explore agent, return findings and ask approval, create OpenSpec proposal if approved, implement only after approval

#### Scenario: Parallel independent work
- **WHEN** adding features in separate modules
- **THEN** CLAUDE.md documents pattern: spawn parallel agents for each module with examples

### Requirement: Git Workflow Conventions
The project SHALL document git workflow conventions in CLAUDE.md for consistent repository practices.

#### Scenario: Branch naming convention
- **WHEN** creating a new branch
- **THEN** CLAUDE.md specifies naming: feature/<name>, fix/<name>, refactor/<name>, docs/<name>

#### Scenario: Commit granularity
- **WHEN** committing changes
- **THEN** CLAUDE.md specifies: one commit per logical change, tests separate or with implementation, documentation separate from code

#### Scenario: Push timing rules
- **WHEN** deciding whether to push
- **THEN** CLAUDE.md specifies: push after completed tasks, after fixing CI failures, before requesting review, never push broken tests or failing lints

### Requirement: Agent Model Selection Guidance
The project SHALL document in CLAUDE.md when to use Haiku vs Sonnet vs Opus for agent spawning.

#### Scenario: Use Haiku for simple tasks
- **WHEN** spawning agents for exploration, file searches, running tests, simple refactoring, or documentation updates
- **THEN** CLAUDE.md indicates to use Haiku model for speed and cost

#### Scenario: Use Sonnet for implementation
- **WHEN** spawning agents for feature implementation, bug fixes, OpenSpec proposals, code reviews, or architecture decisions
- **THEN** CLAUDE.md indicates to use Sonnet model as the default

#### Scenario: Use Opus for complex problems
- **WHEN** spawning agents for complex architectural changes affecting multiple systems or difficult debugging after Sonnet fails
- **THEN** CLAUDE.md indicates to use Opus only when explicitly requested or Sonnet is insufficient

### Requirement: Interactive Slash Commands
Slash commands for task capture SHALL use interactive prompts for structured input.

#### Scenario: Interactive TODO capture
- **WHEN** the /todo command is executed
- **THEN** the command prompts for priority (high/medium/low) and task description before adding to TODO.md

#### Scenario: Interactive idea capture
- **WHEN** the /idea command is executed
- **THEN** the command prompts for category and description before adding to IDEAS.md with timestamp and context

#### Scenario: Non-interactive commands
- **WHEN** /check, /status, /sync, or /clean commands are executed
- **THEN** commands execute immediately without prompts (fixed behavior)

### Requirement: Slash Command Error Handling
Slash commands SHALL handle errors gracefully and provide helpful feedback.

#### Scenario: Command execution failure
- **WHEN** a slash command subprocess fails (non-zero exit code)
- **THEN** the command displays the error output and exits with appropriate error code

#### Scenario: Missing dependencies
- **WHEN** a slash command requires tools not installed (e.g., uv not in PATH)
- **THEN** the command displays clear error message indicating what's missing

#### Scenario: Invalid input in interactive commands
- **WHEN** interactive command receives invalid input (e.g., priority not high/medium/low)
- **THEN** the command re-prompts or provides default value with clear feedback

