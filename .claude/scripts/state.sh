#!/bin/bash
# Fast sitrep: optimized for token cost and speed
# Uses git cache, parallel operations, structured output
# Usage: sitrep-fast [--json] [--no-ci] [--no-openspec] [--refresh]

set -e

source ~/.claude/scripts/git-cache.sh
source ~/.claude/scripts/ci-cache.sh

OUTPUT_FORMAT="text"  # text or json
INCLUDE_CI=true
INCLUDE_OPENSPEC=true
REFRESH_CACHE=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --json) OUTPUT_FORMAT="json" ;;
        --no-ci) INCLUDE_CI=false ;;
        --no-openspec) INCLUDE_OPENSPEC=false ;;
        --refresh) REFRESH_CACHE=true ;;
        --fast) ;; # Default behavior
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

# Invalidate caches if refresh requested
if $REFRESH_CACHE; then
    git_cache_invalidate
    ci_cache_invalidate
fi

# Collect data in parallel
declare -A data

# Git operations (cached)
data[status]=$(git_cache_status)
data[log]=$(git_cache_log)
data[branch]=$(git_cache_branch)
data[current_branch]=$(git_cache_current_branch)

# CI operations (cached)
if $INCLUDE_CI; then
    data[ci_runs]=$(ci_cache_runs)
    data[ci_age]=$(ci_cache_runs_age)
fi

# OpenSpec operations (cached)
if $INCLUDE_OPENSPEC; then
    data[openspec_list]=$(openspec_cache_list)
    data[openspec_age]=$(openspec_cache_list_age)
fi

# Output in chosen format
case $OUTPUT_FORMAT in
    json)
        # Structured output for programmatic consumption
        {
            echo "{"
            echo "  \"git\": {"
            echo "    \"branch\": \"$(echo "${data[current_branch]}" | tr -d '\n')\","
            echo "    \"status\": \"$(echo "${data[status]}" | wc -l) changes\","
            echo "    \"recent_commits\": \"$(echo "${data[log]}" | head -1 | tr -d '"')\""
            echo "  },"

            if $INCLUDE_CI; then
                echo "  \"ci\": {"
                echo "    \"last_run\": \"$(echo "${data[ci_runs]}" | head -2 | tail -1 | awk '{print $NF}' || echo 'unknown')\","
                echo "    \"cache_age\": \"${data[ci_age]}\""
                echo "  },"
            fi

            if $INCLUDE_OPENSPEC; then
                proposal_count=$(echo "${data[openspec_list]}" | grep -c "^[A-Z]" 2>/dev/null || echo 0)
                echo "  \"openspec\": {"
                echo "    \"proposals\": $proposal_count,"
                echo "    \"cache_age\": \"${data[openspec_age]}\""
                echo "  }"
            fi

            echo "}"
        }
        ;;
    *)
        # Human-readable compact format
        echo "Branch: ${data[current_branch]}"
        echo "Status: $(echo "${data[status]}" | wc -l) changes"
        echo "Latest: $(echo "${data[log]}" | head -1)"

        if $INCLUDE_CI; then
            echo "CI: $(echo "${data[ci_runs]}" | head -2 | tail -1 || echo 'unknown') (cached ${data[ci_age]})"
        fi

        if $INCLUDE_OPENSPEC; then
            complete=$(echo "${data[openspec_list]}" | grep "✓" 2>/dev/null | wc -l)
            echo "OpenSpec: $complete complete (cached ${data[openspec_age]})"
        fi
        ;;
esac
