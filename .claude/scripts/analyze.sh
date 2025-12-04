#!/bin/bash
# Analyze audio files → data/generated/

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Error: No audio files specified" >&2
    echo "Usage: /analyze <file1> [file2 ...]" >&2
    exit 1
fi

# Function to sanitize filename for YAML output
sanitize_name() {
    local name="$1"
    # Remove path, keep only basename
    name=$(basename "$name")
    # Remove extension
    name="${name%.*}"
    # Replace spaces and special chars with underscores
    name=$(echo "$name" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    echo "$name"
}

mkdir -p data/generated

for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file" >&2
        continue
    fi

    name=$(sanitize_name "$file")
    output="data/generated/${name}.yaml"

    echo "Analyzing: $(basename "$file") → $output"
    uv run edm analyze --annotations --offline "$file" -o "$output"
done

echo "Analysis complete. Results in data/generated/"
