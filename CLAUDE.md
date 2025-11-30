<!-- OPENSPEC:START -->
# OpenSpec Instructions

OpenSpec provides structured change management for this project. See `@/openspec/AGENTS.md` for workflow details.

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

# OpenSpec Conventions

- Each proposal has a short, one-word SHORTNAME representing its core purpose
- Prefix proposal filenames/titles with SHORTNAME, e.g. [SHORTNAME]name
- Refer to proposals by SHORTNAME
- Commit code after completing any openspec proposal if checks are passing

# Project Context

EDM is an ML-powered system for comprehensive musical and structural analysis of EDM tracks, focusing on BPM detection, beat grids, structural elements (drops, builds, breakdowns), and DJ-specific features (mix points, loop regions).

**Tech Stack:** Python 3.10+, librosa, beat_this (ISMIR 2024 neural beat tracker), essentia, PyTorch/TensorFlow

**Key Paths:**
- Music for testing: `~/music`
- Analysis modules: `src/edm/analysis/`
- Evaluation: `src/edm/evaluation/`
- Tests: `tests/unit/`, `tests/integration/`

# Development

## Context
- Run CLI via uv: `uv run edm ...`
- Always use flags: `--no-color --log-level debug`
- Common tasks: `just` (see `just --list`)
- Pre-commit hooks: Auto-format, lint, and type check on commit
- See `README.md` for installation

## Design Constraints
- **No backward compatibility**: Replace old interfaces directly. No shims or dual interfaces.

# Documentation

See `docs/agent-guide.md` for navigation to architecture, CLI reference, development, testing, and style guides.

# Quality Checkpoints

## Before Committing
- [ ] Run `/check` (tests, types, linting, formatting)
- [ ] Updated relevant docs if architecture/CLI changed
- [ ] No commented-out code remains

## Before Creating OpenSpec Proposal
- [ ] Read existing specs (`openspec list --specs`)
- [ ] Read `openspec/project.md` for conventions
- [ ] Validate with `openspec validate --strict`

# Task Management

Use OpenSpec (`/propose`, `/apply`, `/archive`) for new features, breaking changes, architecture changes, and performance/security work.
