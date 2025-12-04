---
description: Compare analysis accuracy vs annotations
model: haiku
allowed-tools: Bash, Glob
arguments:
  - name: description
    description: File description or glob pattern (optional)
---

Convert the verbal description "$ARGUMENTS.description" to a glob pattern and evaluate the matching annotation files.

Examples:
- "3lau tracks" → data/annotations/*3lau*.yaml
- "falling" → data/annotations/*falling*.yaml
- "dnmo tracks" → data/annotations/*dnmo*.yaml
- Already a path/glob → use as-is
- Empty → evaluate all in data/annotations/

After converting to a glob pattern, expand it and pass all matching files to:
```bash
.claude/scripts/evaluate.sh <expanded files or empty for all>
```

<!--
TEST CASES:
- `/evaluate` → evaluates all tracks in data/annotations/
- `/evaluate 3lau tracks` → finds data/annotations/*3lau*.yaml
- `/evaluate falling` → finds data/annotations/*falling*.yaml
-->
