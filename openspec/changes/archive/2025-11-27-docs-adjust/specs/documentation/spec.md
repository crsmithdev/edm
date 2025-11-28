# Documentation Alignment Specification

## Overview

Ensures documentation and configuration accurately reflect the current codebase implementation.

---

## MODIFIED Requirements

### Requirement: Python Version Requirement

The project SHALL document minimum Python version as 3.12+ consistently across pyproject.toml, README.md, and openspec/project.md.

#### Scenario: pyproject.toml specifies Python 3.12+

- **When** user installs package from pyproject.toml
- **Then** Field shows `requires-python = ">=3.12"` and classifiers include Python 3.12+ only (not 3.10, 3.11)

#### Scenario: README documents Python 3.12+ requirement

- **When** new user reads README.md
- **Then** documentation states "Python 3.12+" as minimum requirement

#### Scenario: openspec/project.md reflects 3.12 baseline

- **When** agent reads project conventions
- **Then** project.md states "Python 3.12+" as development baseline

---

### Requirement: Code Style Configuration Documentation

The Python style guide SHALL accurately document the configured line length of 100 characters (not 88).

#### Scenario: python-style.md documents 100-char line length

- **When** developer reads docs/python-style.md
- **Then** document states line length is 100 characters, not 88

#### Scenario: Configuration reference matches tool setup

- **When** developer checks pyproject.toml for line length configuration
- **Then** both tool.black.line-length and tool.ruff.line-length equal 100

---

### Requirement: Default Log Level Documentation

The CLI documentation SHALL accurately document WARNING as the default log level (not INFO).

#### Scenario: cli-reference.md documents WARNING default

- **When** user reads docs/cli-reference.md log level section
- **Then** default value is documented as "WARNING"

#### Scenario: Main CLI code reflects default

- **When** user checks src/cli/main.py
- **Then** typer.Option default for --log-level is "WARNING"

---

### Requirement: Beat Detection Library Documentation

The documentation SHALL accurately reference beat_this as the primary BPM detection library, not madmom.

#### Scenario: Architecture docs reference beat_this

- **When** developer reads docs/architecture.md
- **Then** documentation states beat_this is used, not madmom

#### Scenario: Development guide references beat_this installation

- **When** developer reads docs/development.md for dependency installation
- **Then** installation instructions reference beat_this, not madmom

#### Scenario: Config parameter documentation clarifies beat_this

- **When** developer reads config.py docstrings for use_madmom parameter
- **Then** docstring clarifies that despite the name, it controls beat_this library (legacy naming)

#### Scenario: BPM detector code clarifies library usage

- **When** developer reads src/edm/analysis/bpm_detector.py function documentation
- **Then** docstrings explicitly state beat_this is used

---

### Requirement: Unimplemented Feature Clarity

The code and documentation SHALL clearly indicate which features are placeholders/unimplemented.

#### Scenario: Structure analysis is documented as unimplemented

- **When** developer reads analyze_structure() docstring and comments
- **Then** documentation explicitly states "returns hardcoded placeholder sections, not actual analysis"

#### Scenario: Beatport integration is documented as unimplemented

- **When** developer reads beatport module docstring
- **Then** documentation states "not yet implemented"

#### Scenario: TuneBat integration is documented as unimplemented

- **When** developer reads tunebat module docstring
- **Then** documentation states "not yet implemented"

#### Scenario: TOML configuration is documented as not functional

- **When** developer reads docs/cli-reference.md or config.py
- **Then** documentation states TOML loading is not yet implemented

#### Scenario: Architecture documentation notes unimplemented features

- **When** user reads docs/architecture.md
- **Then** documentation includes explicit note that structure analysis and external services are placeholder/unimplemented

---

## ADDED Requirements

### Requirement: Line Number Reference Accuracy

All function references in documentation SHALL be accurate to within ±1 line.

#### Scenario: agent-guide.md function references are accurate

- **When** documentation references function locations in agent-guide.md
- **Then** all references match actual line numbers in source files (±1 line tolerance)

#### Scenario: architecture.md function references are accurate

- **When** architecture doc references function locations
- **Then** all references match actual line numbers in source files (±1 line tolerance)

---

## Related Capabilities

- **BPM Detection**: Impacts documentation accuracy for beat_this library usage
- **CLI Interface**: Impacts default log level documentation
- **Configuration**: Impacts Python version requirements and TOML config clarity
