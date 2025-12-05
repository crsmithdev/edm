---
status: deployed
created: 2025-12-05
---

# [FIXEVAL] Fix Evaluation Command Schema Mismatch

## Why

Evaluation command completely broken - expects old schema (`file:` at root) but annotations use new schema (`audio.file`, `structure:` list).

**Impact**: Core evaluation feature non-functional, cannot measure model accuracy.

## What

- `src/edm/evaluation/loader.py` or equivalent - Update `load_yaml_annotations()`
- Handle both old and new schemas for backward compatibility

## Impact

None breaking. Fixes broken feature.
