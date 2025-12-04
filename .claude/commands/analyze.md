---
description: Analyze audio files → data/generated/
allowed-tools: Bash, Glob
arguments:
  - name: description
    description: File description or glob pattern
---

Convert the verbal description "$ARGUMENTS.description" to a glob pattern and analyze the matching audio files.

Examples:
- "3lau tracks" → ~/music/*3lau*.{flac,mp3,wav}
- "falling" → ~/music/*falling*.{flac,mp3,wav}
- "dnmo tracks in music" → ~/music/*dnmo*.{flac,mp3,wav}
- Already a path/glob → use as-is

After converting to a glob pattern, expand it and pass all matching files to:
```bash
.claude/scripts/analyze.sh <expanded files>
```

<!--
TEST CASES:
- `/analyze ~/music/test.flac` → analyzes file directly
- `/analyze 3lau tracks` → finds ~/music/*3lau*.{flac,mp3,wav}
- `/analyze falling` → finds ~/music/*falling*.{flac,mp3,wav}
-->
