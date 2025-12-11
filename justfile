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

# Run annotator web app (React frontend + Flask API)
# Starts both servers: Backend on :5000, Frontend on :5173
# First time: uv sync && cd packages/edm-annotator/frontend && npm install
annotator:
    cd packages/edm-annotator && ./run-dev.sh

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

# Train model (quick test: 10 epochs, small batch)
train-quick:
    uv run edm train data/annotations \
        --audio-dir ~/music \
        --epochs 10 \
        --batch-size 2 \
        --duration 15.0 \
        --output outputs/test_run \
        --run-name quick_test

# Train model (standard: 50 epochs, MERT-95M)
train-standard:
    uv run edm train data/annotations \
        --audio-dir ~/music \
        --epochs 50 \
        --batch-size 4 \
        --backbone mert-95m \
        --output outputs/training \
        --boundary-head \
        --beat-head \
        --energy-head

# Train model (full: 100 epochs, MERT-330M, large batch)
train-full:
    uv run edm train data/annotations \
        --audio-dir ~/music \
        --epochs 100 \
        --batch-size 8 \
        --backbone mert-330m \
        --output outputs/training \
        --boundary-head \
        --beat-head \
        --energy-head

# Train using config file
train-config CONFIG:
    uv run edm train --config {{CONFIG}}

# Resume training from checkpoint
train-resume CHECKPOINT:
    uv run edm train data/annotations \
        --audio-dir ~/music \
        --resume {{CHECKPOINT}} \
        --epochs 100
