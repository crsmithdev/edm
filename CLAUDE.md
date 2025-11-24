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

- Music files for testing are available in `/mnt/c/music/library/`
- Dependencies listed in `pyproject.toml` should never be installed by name on the command line
- Install dev dependencies by default:  `pip install -e ".[dev]"`
- Use a virtual environment and ensure it is activated (`source .venv/bin/activate`)
- Always run the CLI with the `--no-color` and `--verbose` flags

## Interaction
- Be direct and concise - no preambles, politeness padding, apologies or verbosity
- Assume I have staff-level engineering experience

## Source Control
- Use very terse commit messages and put all commit details in the subect by default
- Only include a message body if exceeding > 100 characters in the subject