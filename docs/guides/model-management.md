# Model Management

Complete guide to managing experiments, versioning models, and tracking training runs using DVC and MLflow.

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [DVC: Data & Model Versioning](#dvc-data--model-versioning)
- [MLflow: Experiment Tracking](#mlflow-experiment-tracking)
- [Integration Workflow](#integration-workflow)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

EDM uses two complementary tools for model management:

**DVC (Data Version Control)**:
- Version large files (datasets, checkpoints)
- Selective tracking (only experiments worth keeping)
- Git-based workflow for reproducibility

**MLflow**:
- Automatic experiment logging
- Model registry for production workflows
- Metrics comparison and visualization

### When to Use What

| Task | Tool | Why |
|------|------|-----|
| Track training dataset | DVC | Large files, shared across experiments |
| Version model checkpoint | DVC | Large binary files (100MB+) |
| Log experiment metrics | MLflow | Automatic, searchable, UI for comparison |
| Promote model to production | MLflow | Stage management (Staging → Production) |
| Share checkpoint with team | DVC | Efficient binary file sharing |
| Compare 10 experiments | MLflow | UI with charts and filtering |

---

## Quick Reference

### DVC Workflow

```bash
# Train model
uv run edm train data/annotations --run-name experiment_name

# Version the run
dvc add outputs/training/experiment_name
git add outputs/training/experiment_name.dvc
git commit -m "track experiment: description"

# Push to remote
dvc push outputs/training/experiment_name.dvc

# Retrieve later
dvc pull outputs/training/experiment_name.dvc
```

### MLflow Workflow

```bash
# Start MLflow UI
mlflow ui  # http://localhost:5000

# Training automatically logs to MLflow

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-name edm-training

# Promote model to production
mlflow models update-model-version \
  --name EDMModel \
  --version 3 \
  --stage Production
```

---

## DVC: Data & Model Versioning

### Philosophy

Track selectively, not automatically. Only version experiments worth keeping.

### Directory Structure

```
edm/
├── data/                           # Tracked with data.dvc
│   ├── annotations/
│   └── jams/
├── outputs/                        # Git-ignored
│   └── training/
│       ├── experiment_1/           # Selectively tracked with DVC
│       │   ├── checkpoints/
│       │   ├── logs/
│       │   ├── config.yaml
│       │   └── metadata.yaml
│       ├── experiment_1.dvc        # Committed to git
│       └── temp_run/               # Not tracked (deleted after review)
├── mlruns/                         # MLflow tracking (git-ignored)
├── mlartifacts/                    # MLflow artifacts (git-ignored)
└── .dvc/
    ├── cache/                      # Local storage (git-ignored)
    └── config                      # DVC configuration (committed)
```

### Experiment Lifecycle

#### 1. Run Experiment

```bash
uv run edm train data/annotations \
  --run-name mert95m_baseline \
  --epochs 50 \
  --batch-size 4
```

Output: `outputs/training/mert95m_baseline/`

#### 2. Review Results

```bash
# View TensorBoard
tensorboard --logdir outputs/training/mert95m_baseline/logs

# Check final metrics
cat outputs/training/mert95m_baseline/metadata.yaml

# Inspect configuration
cat outputs/training/mert95m_baseline/config.yaml
```

#### 3. Decision: Keep or Discard

**Keep (track with DVC)**:
```bash
dvc add outputs/training/mert95m_baseline
git add outputs/training/mert95m_baseline.dvc .gitignore
git commit -m "track mert-95m baseline (val_loss=0.0234)"
```

**Discard**:
```bash
rm -rf outputs/training/mert95m_baseline
```

#### 4. Share (Push to Remote)

```bash
# Push to DVC remote (S3/GCS/NAS)
dvc push outputs/training/mert95m_baseline.dvc

# Push git metadata
git push
```

#### 5. Retrieve Later

```bash
# Teammate pulls git metadata
git pull

# See experiment exists
ls outputs/training/*.dvc

# Download checkpoints
dvc pull outputs/training/mert95m_baseline.dvc

# Load checkpoint
python -c "
import torch
ckpt = torch.load('outputs/training/mert95m_baseline/checkpoints/best.pt')
print(f'Epoch: {ckpt[\"epoch\"]}, Val loss: {ckpt[\"best_val_loss\"]:.4f}')
"
```

### Comparing Experiments

**Metadata comparison**:

```bash
# View all experiment metadata
find outputs/training -name "metadata.yaml" -exec echo "---" \; -exec cat {} \;

# Compare specific experiments
diff outputs/training/exp1/metadata.yaml outputs/training/exp2/metadata.yaml
```

**TensorBoard comparison**:

```bash
# Compare multiple runs
tensorboard --logdir outputs/training/

# Open http://localhost:6006
# TensorBoard UI shows all runs side-by-side
```

### Storage Management

**Local cache**:

```bash
# Check cache size
du -sh .dvc/cache

# Remove unused cached files
dvc gc --workspace

# Remove ALL cached files (keeps only committed experiments)
dvc gc --workspace --all-commits
```

**Remote storage**:

```bash
# Add remote (one-time setup)
dvc remote add -d myremote s3://bucket/path
git add .dvc/config
git commit -m "add DVC remote"

# Push to remote
dvc push

# Pull from remote
dvc pull
```

### DVC Remote Options

```bash
# S3
dvc remote add -d s3remote s3://mybucket/dvcstore

# Google Cloud Storage
dvc remote add -d gcs gs://mybucket/dvcstore

# SSH/SCP
dvc remote add -d sshremote ssh://user@example.com/path/to/dvcstore

# Local/NAS
dvc remote add -d nas /mnt/shared/dvcstore
```

For more DVC features, consult the [DVC documentation](https://dvc.org/doc).

---

## MLflow: Experiment Tracking

### Setup

**Installation** (included in dependencies):

```bash
pip install -e .
```

**Start UI**:

```bash
mlflow ui  # Access at http://localhost:5000
```

### Automatic Logging

Training automatically logs to MLflow when using the `Trainer` class:

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
trainer.train()  # Automatically logs to MLflow
```

**Automatically logged**:
- Model checkpoint (`best.pt`)
- Hyperparameters (learning rate, scheduler, etc.)
- Final metrics (best validation loss)
- Git commit hash
- Training configuration

### Manual Logging

For custom workflows:

```python
import mlflow
from pathlib import Path

# Start run
with mlflow.start_run(run_name="custom_experiment"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 4)

    # Log metrics
    for epoch in range(num_epochs):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_model(model, "model")
```

### Model Registry

**Register model**:

```python
from edm.registry import ModelRegistry

registry = ModelRegistry()

# Register trained model
registry.register_model(
    model_path=Path("outputs/training/run_123/checkpoints/best.pt"),
    model_name="EDMModel",
    metadata={
        "boundary_f1": 0.78,
        "training_duration": "2.5 hours",
        "dataset_version": "v1.2"
    }
)
```

**Promote model stages**:

```bash
# List registered models
mlflow models list

# Promote to Staging
mlflow models update-model-version \
  --name EDMModel \
  --version 3 \
  --stage Staging

# Promote to Production
mlflow models update-model-version \
  --name EDMModel \
  --version 3 \
  --stage Production
```

**Load registered model**:

```python
import mlflow.pytorch

# Load latest production model
model = mlflow.pytorch.load_model("models:/EDMModel/Production")

# Load specific version
model = mlflow.pytorch.load_model("models:/EDMModel/3")
```

### Searching and Filtering

**CLI search**:

```bash
# List all experiments
mlflow experiments list

# List runs in experiment
mlflow runs list --experiment-name edm-training

# Search runs with filter
mlflow runs search \
  --experiment-name edm-training \
  --filter "metrics.val_loss < 0.03"

# Order by metric
mlflow runs search \
  --experiment-name edm-training \
  --order-by "metrics.val_loss ASC"
```

**Python API**:

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["edm-training"],
    filter_string="metrics.val_loss < 0.03",
    order_by=["metrics.val_loss ASC"]
)

print(runs[["run_id", "metrics.val_loss", "params.learning_rate"]])
```

### Comparing Runs

**In MLflow UI**:
1. Open http://localhost:5000
2. Select multiple runs (checkboxes)
3. Click "Compare"
4. View parallel coordinates plot, metrics charts, parameter diff

**Programmatically**:

```python
import mlflow
import pandas as pd

# Get runs
runs = mlflow.search_runs(experiment_names=["edm-training"])

# Compare metrics
comparison = runs[["params.learning_rate", "metrics.val_loss", "metrics.boundary_f1"]]
print(comparison.sort_values("metrics.val_loss"))
```

---

## Integration Workflow

### Recommended: DVC + MLflow Together

**Training workflow**:

1. **Train with MLflow tracking** (automatic):
   ```bash
   uv run edm train data/annotations --run-name experiment_1
   ```

2. **Review in MLflow UI**:
   ```bash
   mlflow ui
   # Compare metrics, check if worth keeping
   ```

3. **Version with DVC** (selective):
   ```bash
   # If experiment is worth keeping
   dvc add outputs/training/experiment_1
   git add outputs/training/experiment_1.dvc
   git commit -m "track experiment_1 (val_loss=0.0234)"
   dvc push
   ```

4. **Register in MLflow** (if production-worthy):
   ```python
   registry.register_model(
       model_path=Path("outputs/training/experiment_1/checkpoints/best.pt"),
       model_name="EDMModel"
   )
   ```

### Workflow Summary

```
┌─────────────┐
│   Train     │  → MLflow logs metrics automatically
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Review    │  → MLflow UI: compare experiments
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Decision   │  → Keep or discard?
└──────┬──────┘
       │
       ├─► Keep ──┐
       │          ▼
       │    ┌─────────────┐
       │    │  DVC Track  │  → dvc add, git commit
       │    └─────────────┘
       │
       ├─► Production ──┐
       │                ▼
       │          ┌─────────────┐
       │          │  Register   │  → MLflow model registry
       │          └─────────────┘
       │
       └─► Discard ──► rm -rf
```

---

## Best Practices

### Experiment Naming

**Good names**:
- `mert95m_baseline`
- `cnn_fast_iteration_v2`
- `aug_heavy_dropout0.3`

**Bad names**:
- `test` (not descriptive)
- `run_20231215` (use metadata, not names for dates)
- `final_final_v3` (track versions with DVC, not names)

### Metadata Documentation

Always include in experiment metadata:
- Hypothesis tested
- Dataset version
- Notable configuration changes
- Expected outcome vs. actual

Example `metadata.yaml`:

```yaml
run_name: mert95m_baseline
hypothesis: MERT backbone should outperform CNN by >10%
dataset_version: v1.2 (50 tracks)
config_changes:
  - Increased batch size from 4 to 8
  - Reduced learning rate to 1e-4
results:
  expected_val_loss: <0.03
  actual_val_loss: 0.0287
  conclusion: Hypothesis confirmed
```

### Storage Strategy

**DVC track**:
- ✅ Final experiment checkpoints
- ✅ Best models for comparison
- ✅ Production candidates
- ❌ Intermediate checkpoints
- ❌ Failed experiments
- ❌ Quick tests

**MLflow logs** (automatic):
- ✅ All experiments
- ✅ Intermediate metrics
- ✅ Hyperparameters

### Team Collaboration

1. **Always commit DVC files to git**:
   ```bash
   git add outputs/training/*.dvc
   ```

2. **Push to remote regularly**:
   ```bash
   dvc push
   ```

3. **Pull before training**:
   ```bash
   git pull
   dvc pull data.dvc  # Ensure latest dataset
   ```

4. **Document in commit messages**:
   ```bash
   git commit -m "track mert95m_v3: val_loss=0.0245, boundary_f1=0.81"
   ```

---

## Troubleshooting

### DVC Issues

#### "File already tracked"

```bash
# Error: outputs/training/my_run is already tracked
# Solution: Remove old .dvc file first
rm outputs/training/my_run.dvc
dvc add outputs/training/my_run
```

#### "Cannot find checkpoint"

```bash
# Error: checkpoint file not found
# Solution: Pull from DVC
dvc pull outputs/training/my_run.dvc
```

#### "Cache corrupted"

```bash
# Solution: Clear cache and re-pull
dvc cache clean --force
dvc pull
```

### MLflow Issues

#### "No models found"

```bash
# Check experiment exists
mlflow experiments list

# Create if missing (automatically created on first use)
from edm.registry import ModelRegistry
registry = ModelRegistry()  # Creates "edm-training" experiment
```

#### "Run not found"

Verify tracking URI matches:

```bash
python -c "import mlflow; print(mlflow.get_tracking_uri())"
# Should be: file:///path/to/edm/mlruns
```

#### MLflow UI not showing runs

Start UI from correct directory:

```bash
cd /path/to/edm
mlflow ui --port 5001
# Open http://localhost:5001
```

### Integration Issues

#### DVC and MLflow out of sync

**Symptom**: DVC has checkpoint but MLflow has no corresponding run

**Solution**: Ensure training completed successfully before DVC tracking

```bash
# Check MLflow first
mlflow runs list --experiment-name edm-training

# Then track with DVC
dvc add outputs/training/experiment_name
```

For more troubleshooting, see [Troubleshooting Guide](../reference/troubleshooting.md#experiment-tracking).

---

## See Also

- **[Training Guide](training.md)** - Model training workflows
- **[Deployment Guide](../deployment.md)** - Production deployment
- **[Troubleshooting](../reference/troubleshooting.md)** - Common issues
- **External**:
  - [DVC Documentation](https://dvc.org/doc)
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
