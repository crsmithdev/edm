---
name: Annotation Workflow
description: Generate, merge, and save track annotations
category: Annotations
tags: [annotations, workflow]
arguments:
  - name: command
    description: "Command: generate [files], merge, save, help"
    required: true
---

Run the annotation workflow:

```bash
.claude/scripts/annotate.sh $ARGUMENTS.command
```

**Commands:**
- `generate ~/music/*.flac` - Analyze audio → generated/
- `generate` (no args) - Re-analyze files from reference/
- `merge` - Merge reference + generated → working/
- `save` - Save working → reference/

**Workflow:**
1. `/annotate generate ~/music/*.flac`
2. `/annotate merge`
3. Edit files in `data/annotations/working/`
4. `/annotate save`
5. `/evaluate`
