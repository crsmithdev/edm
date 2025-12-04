---
description: Merge generated annotations → data/annotations/
model: haiku
allowed-tools: Bash
---

```bash
.claude/scripts/annotate.sh
```

<!--
TEST CASES:
- `/annotate` after `/analyze` → merges generated/ into annotations/
- `/annotate` with no generated/ → graceful error
- `/annotate` preserves user edits to annotations/
-->
