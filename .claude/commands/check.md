---
description: Run all quality checks (format, lint, type check, tests) in parallel
model: haiku
allowed-tools: Bash
---

```bash
just check
```

<!--
TEST CASES:
- `/check` → runs format, lint, typecheck, tests in parallel via justfile
-->
