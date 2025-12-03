#!/bin/bash
# Unified annotation workflow script
# Usage: annotate.sh <command> [args...]
#   generate [files...]  - Analyze audio files → generated/
#                          No args: re-analyze files from reference/
#   merge                - Merge reference + generated → working/
#   save                 - Save working → reference/ (strips raw events)
#   edit <track>         - Merge + open in VS Code

set -e

BASE_DIR="/home/crsmi/edm/data/annotations"
REFERENCE_DIR="$BASE_DIR/reference"
GENERATED_DIR="$BASE_DIR/generated"
WORKING_DIR="$BASE_DIR/working"

RAW_START="# --- Raw detected events (original analysis, do not edit) ---"
RAW_END="# --- End raw events ---"

mkdir -p "$REFERENCE_DIR" "$GENERATED_DIR" "$WORKING_DIR"

# Sanitize filename: lowercase, replace special chars with hyphens
sanitize_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:][:punct:]' '-' | tr -cd '[:alnum:]-' | sed 's/^-//;s/-$//'
}

cmd_generate() {
    local files=("$@")

    # No args: get file paths from reference/*.yaml
    if [ ${#files[@]} -eq 0 ]; then
        echo "No files specified, re-analyzing from reference annotations..."
        for ref_file in "$REFERENCE_DIR"/*.yaml; do
            if [ -f "$ref_file" ]; then
                # Extract 'file:' field from YAML
                local audio_path=$(grep -m1 '^file:' "$ref_file" | sed 's/^file:[[:space:]]*//')
                if [ -n "$audio_path" ] && [ -f "$audio_path" ]; then
                    files+=("$audio_path")
                else
                    echo "  [SKIP] File not found: $audio_path (from $(basename "$ref_file"))"
                fi
            fi
        done
    fi

    if [ ${#files[@]} -eq 0 ]; then
        echo "No files to analyze."
        echo "Usage: annotate.sh generate [files...]"
        echo "       annotate.sh generate  (re-analyze from reference/)"
        exit 1
    fi

    echo "Generating annotations for ${#files[@]} file(s)..."
    echo "Output: $GENERATED_DIR"
    echo ""

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            local name=$(sanitize_name "${basename%.*}")
            local output="$GENERATED_DIR/${name}.yaml"

            echo "Analyzing: $basename"
            echo "  -> $output"

            uv run edm analyze --annotations "$file" -o "$output" 2>/dev/null || {
                echo "  [ERROR] Failed to analyze $file"
                continue
            }
        else
            echo "Skipping (not a file): $file"
        fi
    done

    echo ""
    echo "Done. Run '/annotate merge' to prepare for editing."
}

cmd_merge() {
    echo "Merging annotations..."

    # Collect all track names from both generated/ and reference/
    declare -A tracks
    for f in "$GENERATED_DIR"/*.yaml "$REFERENCE_DIR"/*.yaml; do
        [ -f "$f" ] && tracks["$(basename "$f" .yaml)"]=1
    done

    if [ ${#tracks[@]} -eq 0 ]; then
        echo "No tracks found in generated/ or reference/"
        exit 1
    fi

    for name in "${!tracks[@]}"; do
        local ref_file="$REFERENCE_DIR/${name}.yaml"
        local gen_file="$GENERATED_DIR/${name}.yaml"
        local work_file="$WORKING_DIR/${name}.yaml"

        # Extract raw events from generated (if exists)
        local raw_events=""
        if [ -f "$gen_file" ] && grep -q "$RAW_START" "$gen_file" 2>/dev/null; then
            raw_events=$(sed -n "/$RAW_START/,/$RAW_END/p" "$gen_file")
        fi

        if [ -f "$ref_file" ]; then
            echo "  [MERGE] $name (reference + raw events)"
            # Strip existing raw events from reference
            sed "/$RAW_START/,/$RAW_END/d" "$ref_file" | sed '/^---$/d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > "$work_file"
        elif [ -f "$gen_file" ]; then
            echo "  [NEW] $name (from generated)"
            # Use generated but strip raw events first
            sed "/$RAW_START/,/$RAW_END/d" "$gen_file" | sed '/^---$/d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > "$work_file"
        else
            continue
        fi

        # Append raw events if available
        if [ -n "$raw_events" ]; then
            echo "" >> "$work_file"
            echo "---" >> "$work_file"
            echo "$raw_events" >> "$work_file"
        fi
    done

    echo ""
    echo "Working files ready in: $WORKING_DIR"
    echo "Edit there, then run '/annotate save'"
}

cmd_save() {
    echo "Saving annotations to reference..."

    local count=0
    for work_file in "$WORKING_DIR"/*.yaml; do
        if [ -f "$work_file" ]; then
            local name=$(basename "$work_file" .yaml)
            local ref_file="$REFERENCE_DIR/${name}.yaml"

            echo "  [SAVE] $name"
            # Extract content before --- separator
            sed '/^---$/,$d' "$work_file" | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > "$ref_file"
            echo "" >> "$ref_file"
            ((count++))
        fi
    done

    if [ $count -eq 0 ]; then
        echo "No working files to save."
        exit 1
    fi

    echo ""
    echo "Saved $count file(s) to: $REFERENCE_DIR"
    echo "Run '/evaluate' to measure accuracy."
}

cmd_help() {
    echo "Annotation workflow:"
    echo ""
    echo "  generate [files]  Analyze audio files → generated/"
    echo "                    No args: re-analyze files from reference/"
    echo "  merge             Merge reference + generated → working/"
    echo "  save              Save working → reference/ (strips raw)"
    echo ""
    echo "Workflow:"
    echo "  1. /annotate generate ~/music/*.flac"
    echo "  2. /annotate merge"
    echo "  3. Edit files in data/annotations/working/"
    echo "  4. /annotate save"
    echo "  5. /evaluate"
    echo ""
    echo "Directories:"
    echo "  reference/ - Your annotations (git-tracked)"
    echo "  generated/ - Machine output (disposable)"
    echo "  working/   - Edit workspace (disposable)"
}

# Main dispatch
case "${1:-help}" in
    generate|gen|g)
        shift
        cmd_generate "$@"
        ;;
    merge|m)
        cmd_merge
        ;;
    save|s)
        cmd_save
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
