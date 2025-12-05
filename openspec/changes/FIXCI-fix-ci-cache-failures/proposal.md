---
status: ready
created: 2025-12-05
---

# [FIXCI] Fix CI Cache Failures (50% Failure Rate)

## Why

GitHub Actions cache service experiencing infrastructure failures causing 50% CI failure rate.

**Errors**: `Cache service responded with 400`, `Our services aren't available right now`

**Impact**: Blocking merges, wasted compute, developer friction.

## What

- `.github/workflows/ci.yml` lines 22-25, 60-63
- Disable `enable-cache: true` in `astral-sh/setup-uv@v4`

## Impact

None breaking. Slight CI slowdown (no cache) but eliminates failures.
