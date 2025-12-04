#!/bin/bash
# Evaluate analysis accuracy

set -euo pipefail

# Default to data/annotations/ directory
if [ $# -eq 0 ]; then
    uv run edm evaluate --reference data/annotations/
else
    # Assume args are directory paths
    uv run edm evaluate --reference "$@"
fi
