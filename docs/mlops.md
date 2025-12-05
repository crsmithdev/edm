# MLOps: Model Registry & Tracking

## Overview

EDM uses MLflow for model versioning, experiment tracking, and model registry operations. This enables reproducible training, model promotion workflows, and production deployment.

## MLflow Setup

### Installation

MLflow is included in the project dependencies:

```bash
pip install -e .
```

### Directory Structure

```
edm/
├── mlruns/          # MLflow tracking data (gitignored)
├── mlartifacts/     # Model artifacts (gitignored)
└── outputs/         # Training outputs (DVC-tracked)
```

### Starting the UI

View experiments and models:

```bash
mlflow ui
```

Access at http://localhost:5000

## Model Registry

### Architecture

The `ModelRegistry` class in `src/edm/registry/mlflow_registry.py` provides:

- **Automatic logging**: Integrates with `Trainer` to log models after training
- **Version management**: Track all model versions with metadata
- **Stage promotion**: Promote models through Staging → Production workflow
- **Artifact tracking**: Store model checkpoints with run metadata

### Usage in Training

```python
from edm.registry import ModelRegistry
from edm.training import Trainer, TrainingConfig

# Initialize registry
registry = ModelRegistry(experiment_name="edm-training")

# Create trainer with registry
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    registry=registry,  # Enable automatic logging
)

# Train model
trainer.train()  # Automatically logs to MLflow on completion
```

The trainer will automatically log:
- Model checkpoint (`best.pt`)
- Hyperparameters (learning rate, scheduler, etc.)
- Final metrics (best validation loss)
- Git commit hash
- Training configuration

### Manual Logging

For custom workflows:

```python
registry = ModelRegistry()

registry.log_model(
    model_path=Path("outputs/run_20251205/best.pt"),
    run_name="mert-95m-experiment",
    params={
        "backbone": "mert-95m",
        "learning_rate": 1e-4,
        "num_epochs": 50,
    },
    metrics={
        "best_val_loss": 0.234,
    },
    tags={
        "experiment": "baseline",
    }
)
```

## CLI Commands

### List Models

```bash
edm models list
```

Shows registered models with:
- Version number
- Current stage
- Run name
- Validation loss
- Backbone architecture
- Creation timestamp

Options:
- `--max-results N`: Limit to N models (default: 10)

### Model Info

```bash
edm models info 3
```

Detailed information for version 3:
- Stage and run metadata
- All metrics
- All hyperparameters
- Tags (git commit, experiment name, etc.)

### Promote Model

```bash
edm models promote 3 Production
```

Promote model version 3 to Production stage.

**Workflow**:
1. Train model → Automatically logged (None stage)
2. `edm models promote 5 Staging` → Stage for testing
3. Test in staging environment
4. `edm models promote 5 Production` → Deploy to production

**Stages**:
- `Staging`: Testing/validation
- `Production`: Active deployment
- `Archived`: Deprecated models

### Load Production Model

```bash
edm models load-production
```

Downloads current production model to `./downloads/`.

## Experiment Tracking

### Logged Data

Each training run logs:

**Parameters**:
- `backbone`: Model architecture (mert-95m, mert-330m, etc.)
- `num_epochs`: Training duration
- `learning_rate`: Initial LR
- `scheduler`: LR schedule type
- `weight_decay`: L2 regularization
- `gradient_clip`: Max gradient norm

**Metrics**:
- `best_val_loss`: Best validation loss
- `best_epoch`: Epoch with best loss

**Tags**:
- `git_commit`: Code version
- `model_type`: "multitask"

**Artifacts**:
- Model checkpoint (`.pt` file)

### Comparing Experiments

In MLflow UI:
1. Navigate to Experiments
2. Select multiple runs
3. Click "Compare"
4. View parameter/metric charts

## Backend Configuration

### SQLite (Default)

Local file-based tracking:

```python
registry = ModelRegistry(tracking_uri="./mlruns")
```

### PostgreSQL (Production)

For multi-user environments:

```python
registry = ModelRegistry(
    tracking_uri="postgresql://user:pass@host:5432/mlflow"
)
```

Set up database:

```sql
CREATE DATABASE mlflow;
```

Start MLflow server:

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@host:5432/mlflow \
    --default-artifact-root s3://my-bucket/mlflow \
    --host 0.0.0.0
```

## Best Practices

### Naming Conventions

**Run names**: Use `generate_run_name()`:
```python
from edm.training import generate_run_name

run_name = generate_run_name(
    backbone="mert-95m",
    description="baseline-lr1e4"
)
# Example: run_20251205_143022_mert-95m_baseline-lr1e4
```

**Tags**: Use consistent experiment tags:
```python
tags = {
    "experiment": "baseline",       # Experiment group
    "dataset_version": "v1.2",     # Data version
    "notes": "Testing new loss fn"
}
```

### Promotion Workflow

1. **Train**: Model logged automatically
2. **Validate**: Review metrics in MLflow UI
3. **Stage**: `edm models promote <v> Staging`
4. **Test**: Run evaluation on held-out test set
5. **Deploy**: `edm models promote <v> Production`
6. **Rollback**: `edm models promote <old_v> Production` if needed

### Versioning

- **Models**: Tracked by MLflow (auto-increment)
- **Code**: Git commit hash logged in tags
- **Data**: DVC hash in metadata (tracked by Trainer)

## Integration with DVC

DVC tracks:
- Training data (`data/`)
- Large model artifacts (`outputs/`)

MLflow tracks:
- Experiments and hyperparameters
- Model registry and promotion
- Small artifacts (best checkpoints)

**Workflow**:
1. DVC version data: `dvc add data`
2. Train with MLflow: `edm train --registry`
3. DVC track outputs: `dvc add outputs/run_*`
4. Promote in MLflow: `edm models promote <v> Production`

## Troubleshooting

### "No models found"

```bash
# Check experiment exists
mlflow experiments list

# Create if missing (automatically created on first use)
from edm.registry import ModelRegistry
registry = ModelRegistry()  # Creates "edm-training" experiment
```

### "Run not found"

Verify tracking URI matches:

```bash
# Check current URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Should be: file:///home/user/edm/mlruns
```

### UI not showing models

Clear browser cache or restart UI:

```bash
mlflow ui --port 5001  # Use different port
```

## API Reference

### ModelRegistry

```python
class ModelRegistry:
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "edm-training",
    ) -> None: ...

    def log_model(
        self,
        model_path: Path,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str: ...

    def promote_model(self, version: int, stage: str) -> None: ...

    def load_production_model(self) -> Path | None: ...

    def list_models(self, max_results: int = 10) -> list[dict[str, Any]]: ...

    def get_model_info(self, version: int) -> dict[str, Any] | None: ...
```

## Next Steps

- **Monitoring**: Set up [Prometheus/Grafana](../openspec/changes/MONITOR-prometheus-grafana-monitoring/proposal.md)
- **Deployment**: Implement [FastAPI inference service](../openspec/changes/SERVE-fastapi-inference-service/proposal.md)
- **A/B Testing**: Compare Production models with staged candidates
