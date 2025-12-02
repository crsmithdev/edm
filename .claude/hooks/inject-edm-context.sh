#!/bin/bash
# Context-aware prompt injection for Claude Code
# Detects task category from stdin and injects relevant XML context

cd "$(dirname "$0")/../.." || exit 0

# Read user's prompt from stdin
PROMPT=$(cat)

# Only inject if in EDM project
if [[ ! -f "pyproject.toml" ]] || ! grep -q "edm" pyproject.toml 2>/dev/null; then
    echo "$PROMPT"
    exit 0
fi

CONTEXTS_DIR="$(dirname "$0")/../contexts"
INJECT_FILES=()

# Category detection (case-insensitive)
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')

# OpenSpec keywords: proposal, spec, workflow, apply, archive, validate
if echo "$PROMPT_LOWER" | grep -qE '(proposal|openspec|spec|workflow|apply|archive|validate|design\.md|tasks\.md)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/openspec.xml")
fi

# Audio analysis keywords: bpm, beat, structure, drop, breakdown, msaf, librosa, beat_this
if echo "$PROMPT_LOWER" | grep -qE '(bpm|beat|tempo|structure|drop|breakdown|buildup|intro|outro|msaf|librosa|beat_this|essentia|detector|section|bar)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/audio.xml")
fi

# Python/CLI keywords: typer, cli, pydantic, pytest, ruff (always inject for code changes)
if echo "$PROMPT_LOWER" | grep -qE '(typer|cli|command|pydantic|pytest|test|ruff|mypy|import|class|function|def |async|exception)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/python.xml")
fi

# Default: inject audio context for general queries (most common task)
if [ ${#INJECT_FILES[@]} -eq 0 ]; then
    INJECT_FILES+=("$CONTEXTS_DIR/audio.xml")
fi

# Inject contexts in order, then user prompt
for file in "${INJECT_FILES[@]}"; do
    if [ -f "$file" ]; then
        cat "$file"
        echo
    fi
done

echo "$PROMPT"
