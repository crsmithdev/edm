#!/bin/bash
# CI/OpenSpec cache system for expensive operations
# Caches: CI runs, OpenSpec proposals with configurable TTL
# Usage: source ci-cache.sh; ci_cache_runs; openspec_cache_list; etc.

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/claude-ci"
CACHE_TTL=${CI_CACHE_TTL:-300}  # 5 minutes default

mkdir -p "$CACHE_DIR"

# Generate cache key for current repo
ci_cache_key() {
    local repo_root=$(git rev-parse --show-toplevel 2>/dev/null)
    echo "${repo_root//\//_}"
}

# Check if cache is valid (within TTL)
ci_cache_valid() {
    local cache_file="$1"
    if [[ ! -f "$cache_file" ]]; then
        return 1
    fi
    local age=$(($(date +%s) - $(stat -f%m "$cache_file" 2>/dev/null || stat -c%Y "$cache_file")))
    [[ $age -lt $CACHE_TTL ]]
}

# Get cache age in seconds
ci_cache_age() {
    local cache_file="$1"
    if [[ ! -f "$cache_file" ]]; then
        echo "never"
        return
    fi
    local age=$(($(date +%s) - $(stat -f%m "$cache_file" 2>/dev/null || stat -c%Y "$cache_file")))
    echo "${age}s"
}

# Get or compute CI runs
ci_cache_runs() {
    local key=$(ci_cache_key)
    local cache_file="$CACHE_DIR/ci_runs_${key}"

    if ci_cache_valid "$cache_file"; then
        cat "$cache_file"
    else
        gh run list --limit 3 2>/dev/null | tee "$cache_file" || echo "CI: unavailable"
    fi
}

# Get CI cache age
ci_cache_runs_age() {
    local key=$(ci_cache_key)
    local cache_file="$CACHE_DIR/ci_runs_${key}"
    ci_cache_age "$cache_file"
}

# Get or compute OpenSpec list
openspec_cache_list() {
    local key=$(ci_cache_key)
    local cache_file="$CACHE_DIR/openspec_list_${key}"

    if ci_cache_valid "$cache_file"; then
        cat "$cache_file"
    else
        openspec list 2>/dev/null | tee "$cache_file" || echo "OpenSpec: unavailable"
    fi
}

# Get OpenSpec cache age
openspec_cache_list_age() {
    local key=$(ci_cache_key)
    local cache_file="$CACHE_DIR/openspec_list_${key}"
    ci_cache_age "$cache_file"
}

# Invalidate all caches for current repo
ci_cache_invalidate() {
    local key=$(ci_cache_key)
    rm -f "$CACHE_DIR"/*_${key}
}
