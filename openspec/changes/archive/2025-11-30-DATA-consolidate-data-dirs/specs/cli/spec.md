## ADDED Requirements

### Requirement: Evaluate Command
The CLI SHALL provide an `evaluate` subcommand for accuracy evaluation.

#### Scenario: BPM evaluation with default output
- **WHEN** user runs `edm evaluate bpm <reference.csv>`
- **THEN** results are written to `data/accuracy/bpm/`

#### Scenario: Structure evaluation with default output
- **WHEN** user runs `edm evaluate structure <reference.csv> <audio_dir>`
- **THEN** results are written to `data/accuracy/structure/`

#### Scenario: Custom output directory
- **WHEN** user provides `--output <path>`
- **THEN** results are written to specified path instead of default
