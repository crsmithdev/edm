#!/bin/bash
# Inject JavaScript/TypeScript context when JS/TS keywords detected

prompt="$1"

if echo "$prompt" | grep -qiE "javascript|typescript|js|ts|jsx|tsx|eslint|prettier|vitest|node\.js|npm|pnpm|package\.json"; then
    cat "$(dirname "$0")/../contexts/javascript.xml"
fi
