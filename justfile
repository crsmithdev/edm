# EDM task runner

# Default recipe to display help
default:
    @just --list

# Install dependencies
install:
    uv sync

# Clean build artifacts and caches
clean:
    .claude/scripts/clean.sh

# Wipe and reinstall all dependencies
deps:
    .claude/scripts/deps.sh

# Format code
fmt:
    uv run ruff format .

# Lint code with auto-fix
lint:
    uv run ruff check --fix .

# Type check
types:
    uv run mypy packages/edm-lib/src/

# Run tests
test *ARGS='':
    uv run pytest {{ARGS}}

# Run tests with coverage
test-cov:
    uv run pytest --cov=packages/edm-lib/src --cov-report=term --cov-report=html

# Run annotator web app
annotator:
    cd packages/edm-annotator && uv run flask --app src/edm_annotator/app run

# Run all quality checks
check:
    .claude/scripts/check.sh

# Analyze audio file(s)
analyze *ARGS='':
    uv run edm --no-color --log-level debug analyze {{ARGS}}

# Evaluate accuracy
evaluate *ARGS='':
    uv run edm --no-color --log-level debug evaluate {{ARGS}}

# Git status for commit prep
ship:
    ~/.claude/scripts/ship.sh

# Project status overview
sitrep:
    ~/.claude/scripts/sitrep.sh

# List OpenSpec proposals
list:
    openspec list

# Show OpenSpec proposal
show PROPOSAL:
    openspec show {{PROPOSAL}}

# Validate OpenSpec proposal
validate PROPOSAL:
    openspec validate {{PROPOSAL}} --strict
