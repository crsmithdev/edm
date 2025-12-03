---
name: Evaluate Reference
description: Evaluate analysis accuracy against reference annotations
category: Analysis
tags: [evaluation, accuracy, testing]
---

Run evaluation against reference annotations in `data/annotations/reference/`:

```bash
uv run edm evaluate
```

This compares detected structure against your manually-annotated YAML files and outputs accuracy metrics (boundary F1, label accuracy, event precision/recall).

To add reference annotations, copy your verified annotation YAML files to `data/annotations/reference/`.
