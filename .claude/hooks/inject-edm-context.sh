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

# Audio analysis keywords: bpm, beat, structure, drop, breakdown, msaf, librosa, beat_this
if echo "$PROMPT_LOWER" | grep -qE '(bpm|beat|tempo|structure|drop|breakdown|buildup|intro|outro|msaf|librosa|beat_this|essentia|detector|section|bar)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/audio.xml")
fi

# EDM terminology keywords: edm, track structure, song structure, terminology
if echo "$PROMPT_LOWER" | grep -qE '(edm|track.*(structure|terminology|concept)|song.*(structure|terminology)|drop|buildup|breakdown|phrase|downbeat|bar|energy.*(level|dynamic))'; then
    INJECT_FILES+=("$CONTEXTS_DIR/edm-terminology.xml")
fi

# Evaluation keywords: evaluate, accuracy, reference, metrics
if echo "$PROMPT_LOWER" | grep -qE '(evaluat|accuracy|reference|metrics|ground.?truth|f1|precision|recall)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/evaluation.xml")
fi

# Python/CLI keywords: typer, cli, pydantic, pytest, ruff (always inject for code changes)
if echo "$PROMPT_LOWER" | grep -qE '(typer|edm cli|cli command|pydantic|pytest|unit.?test|ruff|mypy|import |class [A-Z]|function|def [a-z]|async def|exception|structlog)'; then
    INJECT_FILES+=("$CONTEXTS_DIR/python.xml")
fi

# Default: inject audio context for general queries (most common task)
if [ ${#INJECT_FILES[@]} -eq 0 ]; then
    INJECT_FILES+=("$CONTEXTS_DIR/audio.xml")
fi

# Report which contexts were injected
if [ ${#INJECT_FILES[@]} -gt 0 ]; then
    NAMES=()
    for file in "${INJECT_FILES[@]}"; do
        NAMES+=("$(basename "$file" .xml)")
    done
    echo "<!-- project-contexts: ${NAMES[*]} -->"
    echo

    # Track contexts for status line display
    if [ -x "$HOME/.claude/scripts/session-tracker.sh" ]; then
        for name in "${NAMES[@]}"; do
            "$HOME/.claude/scripts/session-tracker.sh" add_context "$name" 2>/dev/null || true
        done
    fi
fi

# Inject contexts in order, then user prompt
for file in "${INJECT_FILES[@]}"; do
    if [ -f "$file" ]; then
        cat "$file"
        echo
    fi
done

echo "$PROMPT"
