---
name: Generate Annotations
description: Generate annotation files for audio tracks matching a glob pattern
category: Analysis
tags: [analysis, annotations]
arguments:
  - name: pattern
    description: Glob pattern for audio files (e.g., "~/music/*.flac", "~/music/**/*.mp3")
    required: true
---

Run the annotation generation script with the provided glob pattern:

```bash
.claude/scripts/generate-annotations.sh '$ARGUMENTS.pattern'
```

Output files are written to `data/annotations/generated/` with one YAML file per track.
