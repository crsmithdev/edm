# Development Workflow Spec Delta

## ADDED Requirements

### Requirement: Context Injection

The system SHALL inject task-specific XML context before user prompts via UserPromptSubmit hook.

#### Scenario: OpenSpec task detection

- **WHEN** prompt contains openspec-related keywords (proposal, openspec, spec, change, archive, design.md, tasks.md)
- **THEN** inject openspec.xml context with workflow, spec format, and validation info

#### Scenario: Audio task detection

- **WHEN** prompt contains audio-related keywords (bpm, beat, structure, audio, detector, librosa, analysis, energy, msaf, explore, research, find, where, how)
- **THEN** inject audio.xml context with algorithms, fallback chains, data types, and file layout

#### Scenario: Default Python context

- **WHEN** prompt does not match openspec or audio categories
- **THEN** inject python.xml context with code style, CLI patterns, and testing conventions

### Requirement: Context Documentation References

Context files SHALL reference authoritative documentation rather than duplicating detailed information.

#### Scenario: Algorithm reference in audio context

- **WHEN** audio.xml mentions fallback chains or thresholds
- **THEN** include reference to docs/analysis-algorithms.md for full details

#### Scenario: Architecture reference in audio context

- **WHEN** audio.xml mentions module patterns or caching
- **THEN** include reference to docs/architecture.md for implementation details

#### Scenario: CLI reference in python context

- **WHEN** python.xml mentions Typer or CLI patterns
- **THEN** include reference to docs/cli-patterns.md for conventions

### Requirement: Analysis Algorithm Documentation

The system SHALL maintain documentation of analysis algorithms and their fallback chains.

#### Scenario: Fallback chain documentation

- **WHEN** developer needs to understand BPM detection strategy
- **THEN** docs/analysis-algorithms.md describes beat_this → librosa fallback with conditions

#### Scenario: Threshold documentation

- **WHEN** developer needs to understand structure labeling
- **THEN** docs/analysis-algorithms.md documents energy thresholds (>0.7 = drop, gradient >0.15 = buildup)

### Requirement: CLI Pattern Documentation

The system SHALL maintain documentation of CLI conventions and patterns.

#### Scenario: Command structure documentation

- **WHEN** developer needs to add a new CLI command
- **THEN** docs/cli-patterns.md describes Typer app registration and command structure

#### Scenario: Output formatting documentation

- **WHEN** developer needs to format CLI output
- **THEN** docs/cli-patterns.md describes Rich Console patterns for tables, JSON, YAML
