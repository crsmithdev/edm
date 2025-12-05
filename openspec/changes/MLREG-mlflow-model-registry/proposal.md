---
status: ready
created: 2025-12-05
---

# [MLREG] Implement MLflow Model Registry

## Why

**No Model Versioning**: Models saved as `best.pt` in timestamped directories with no centralized tracking.

**Current Gaps**:
- Cannot answer "which model is in production?"
- No model metadata queryable (accuracy, hyperparameters, dataset)
- No model lineage or promotion workflow
- Manual DVC tracking error-prone

**Impact**: Foundation for all MLOps - enables model deployment, A/B testing, rollback.

## What

- Create `src/edm/registry/mlflow_registry.py` - Model registry wrapper
- Integrate with `src/edm/training/trainer.py` - Auto-log on training completion
- Add `mlflow>=2.10.0` dependency
- SQLite backend initially (production: PostgreSQL)

## Impact

- **Effort**: 3 days
- **ROI**: High - foundational capability
- **Dependencies**: MLflow
- **Migration**: Existing checkpoints can be retroactively registered
