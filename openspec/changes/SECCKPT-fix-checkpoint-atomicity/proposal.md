---
status: ready
created: 2025-12-05
---

# [SECCKPT] Fix Checkpoint Corruption Risk with Atomic Writes

## Why

`src/edm/training/trainer.py:396` writes checkpoints directly with `torch.save()`, risking corruption if process crashes during write.

**Risk**: Lost training progress, corrupted model files requiring restart from scratch.

## What

- `src/edm/training/trainer.py` - Checkpoint saving logic

## Impact

None breaking. Transparent safety improvement.
