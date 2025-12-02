#!/bin/bash
# Generate annotation files for audio tracks
# Usage: generate-annotations.sh <file-or-glob-pattern> [more files...]
# Example: generate-annotations.sh ~/music/*.flac
#          generate-annotations.sh ~/music/**/*.mp3
#          generate-annotations.sh "~/music/Some Track.flac"

set -e

if [ -z "$1" ]; then
    echo "Usage: generate-annotations.sh <file-or-glob-pattern> [more files...]"
    echo "Example: generate-annotations.sh ~/music/*.flac"
    exit 1
fi

OUTPUT_DIR="/home/crsmi/edm/data/annotations/generated"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# All arguments are files (glob already expanded by shell)
FILES=("$@")

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files provided"
    exit 1
fi

echo "Found ${#FILES[@]} file(s) to analyze"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process each file
for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        # Get basename without extension, sanitize for filename
        BASENAME=$(basename "$FILE")
        BASENAME_NO_EXT="${BASENAME%.*}"
        # Convert to lowercase, replace spaces/special chars with hyphens, collapse multiple hyphens
        SAFE_NAME=$(echo "$BASENAME_NO_EXT" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:][:punct:]' '-' | tr -cd '[:alnum:]-' | sed 's/^-//;s/-$//')
        OUTPUT_FILE="$OUTPUT_DIR/${SAFE_NAME}.yaml"

        echo "Analyzing: $BASENAME"
        echo "  -> $OUTPUT_FILE"

        uv run edm analyze --annotations "$FILE" -o "$OUTPUT_FILE" 2>/dev/null || {
            echo "  [ERROR] Failed to analyze $FILE"
            continue
        }
    else
        echo "Skipping (not a file): $FILE"
    fi
done

echo ""
echo "Done. Annotations written to $OUTPUT_DIR"
