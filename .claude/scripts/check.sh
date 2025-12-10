#!/usr/bin/env bash
set -e

# EDM Quality Checks
# Runs all quality checks for both backend and frontend

echo "🔍 Running quality checks..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track failures
FAILED=0

# Backend checks
echo "📦 Backend Checks"
echo "─────────────────"

# Lint
echo -n "  Lint (ruff)... "
if uv run ruff check . &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    FAILED=$((FAILED + 1))
fi

# Type check
echo -n "  Types (mypy)... "
if uv run mypy packages/edm-lib/src/ &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    FAILED=$((FAILED + 1))
fi

echo ""

# Frontend checks
if [ -d "packages/edm-annotator/frontend" ]; then
    echo "🌐 Frontend Checks"
    echo "──────────────────"

    cd packages/edm-annotator/frontend

    # Type check
    echo -n "  Types (tsc)... "
    if npm run type-check &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        FAILED=$((FAILED + 1))
    fi

    # Lint
    echo -n "  Lint (eslint)... "
    if npm run lint &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        FAILED=$((FAILED + 1))
    fi

    cd - &>/dev/null
    echo ""
fi

# Summary
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $FAILED check(s) failed${NC}"
    exit 1
fi
