#!/bin/bash
# Merge data/generated/ → data/annotations/
# Preserves user edits to structure, updates raw data

set -euo pipefail

ANNOTATIONS_DIR="data/annotations"
GENERATED_DIR="data/generated"

RAW_START="# --- Raw detected events (original analysis, do not edit) ---"
RAW_END="# --- End raw events ---"

mkdir -p "$ANNOTATIONS_DIR" "$GENERATED_DIR"

# Check if generated dir has any files
if ! ls "$GENERATED_DIR"/*.yaml >/dev/null 2>&1; then
    echo "No generated annotations found in $GENERATED_DIR/"
    echo "Run /analyze first to generate annotations."
    exit 0
fi

echo "Merging generated annotations → $ANNOTATIONS_DIR/"

for gen_file in "$GENERATED_DIR"/*.yaml; do
    name=$(basename "$gen_file")
    anno_file="$ANNOTATIONS_DIR/$name"

    # Extract raw events from generated
    raw_events=""
    if grep -q "$RAW_START" "$gen_file" 2>/dev/null; then
        raw_events=$(sed -n "/$RAW_START/,/$RAW_END/p" "$gen_file")
    fi

    if [ -f "$anno_file" ]; then
        echo "  [UPDATE] $name (preserving user edits, updating raw data)"
        # Strip old raw events from annotation, keep user data
        sed "/$RAW_START/,/$RAW_END/d" "$anno_file" | sed '/^---$/d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > "${anno_file}.tmp"

        # Append new raw events
        if [ -n "$raw_events" ]; then
            echo "" >> "${anno_file}.tmp"
            echo "---" >> "${anno_file}.tmp"
            echo "$raw_events" >> "${anno_file}.tmp"
        fi

        mv "${anno_file}.tmp" "$anno_file"
    else
        echo "  [NEW] $name"
        # New file: copy from generated
        cp "$gen_file" "$anno_file"
    fi
done

echo ""
echo "Merge complete. Annotations in $ANNOTATIONS_DIR/"
echo "Edit annotations, then run /evaluate to check accuracy."
