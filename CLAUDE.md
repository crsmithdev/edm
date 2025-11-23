<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Development Context

### Music Library Location
Test music files are mounted at: `/mnt/c/music/library`

This directory contains FLAC audio files for testing the EDM analysis CLI.

### CLI Usage
1. Always activate the virtual environment first: `source .venv/bin/activate`
2. Always run the CLI with the `--no-color` option for cleaner output
3. Example: `edm analyze "/mnt/c/music/library/Artbat - Artefact.flac" --offline --ignore-metadata --no-color --verbose`

### Installation
When installing dependencies, always install dev dependencies as well:
```bash
pip install -e ".[dev]"
```