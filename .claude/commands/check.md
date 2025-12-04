---
description: Run all quality checks (format, lint, type check, tests) in parallel
model: haiku
allowed-tools: Bash
---

```bash
~/.claude/scripts/check.sh
```

<!--
TEST CASES:
- `/check` in repo with justfile → runs fmt, lint, types, test targets
- `/check` in repo without justfile → reports just not found, suggests checking repo setup
-->
