# Experiment Tracking

Guide to managing machine learning experiments with DVC and git.

## Overview

The EDM project uses DVC (Data Version Control) to track:
- Training datasets (`data/annotations/` - already tracked)
- Model checkpoints (selective tracking)
- Training runs (selective tracking)

**Philosophy:** Track selectively, not automatically. Only version experiments worth keeping.

## Quick Reference

```bash
# Train model
uv run edm train data/annotations --run-name experiment_name

# Version the run
dvc add outputs/training/experiment_name
git add outputs/training/experiment_name.dvc
git commit -m "track experiment: description"

# Retrieve versioned checkpoint
dvc pull outputs/training/experiment_name.dvc

# List all versioned experiments
ls outputs/training/*.dvc
```

## Directory Structure

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
└── .dvc/
    ├── cache/                      # Local storage (git-ignored)
    └── config                      # DVC configuration (committed)
```

## Workflow: Experiment Lifecycle

### 1. Run Experiment

```bash
uv run edm train data/annotations \
  --run-name mert95m_baseline \
  --epochs 50 \
  --batch-size 4
```

Output: `outputs/training/mert95m_baseline/`

### 2. Review Results

```bash
# View TensorBoard
tensorboard --logdir outputs/training/mert95m_baseline/logs

# Check final metrics
cat outputs/training/mert95m_baseline/metadata.yaml

# Inspect configuration
cat outputs/training/mert95m_baseline/config.yaml
```

### 3. Decision: Keep or Discard

**Keep (track with DVC):**
```bash
dvc add outputs/training/mert95m_baseline
git add outputs/training/mert95m_baseline.dvc .gitignore
git commit -m "track mert-95m baseline (val_loss=0.0234)"
```

**Discard:**
```bash
rm -rf outputs/training/mert95m_baseline
```

### 4. Share (Push to Remote)

```bash
# Push to DVC remote (S3/GCS/NAS)
dvc push outputs/training/mert95m_baseline.dvc

# Push git metadata
git push
```

### 5. Retrieve Later

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

## Comparing Experiments

### Metadata Comparison

```bash
# View all experiment metadata
find outputs/training -name "metadata.yaml" -exec echo "---" \; -exec cat {} \;

# Compare specific experiments
diff outputs/training/exp1/metadata.yaml outputs/training/exp2/metadata.yaml
```

### TensorBoard Comparison

```bash
# Compare multiple runs
tensorboard --logdir outputs/training/

# Open http://localhost:6006
# TensorBoard UI shows all runs side-by-side
```

### Git History

```bash
# See all tracked experiments
git log --oneline --all -- outputs/training/*.dvc

# See when experiment was tracked
git log outputs/training/mert95m_baseline.dvc
```

## Storage Management

### Local Cache

DVC stores files in `.dvc/cache/`:

```bash
# Check cache size
du -sh .dvc/cache

# Remove unused cached files
dvc gc --workspace

# Remove ALL cached files (keeps only committed experiments)
dvc gc --workspace --all-commits
```

### Remote Storage

Configure remote storage once, use everywhere:

```bash
# Add remote
dvc remote add -d myremote s3://bucket/path

# Modify .dvc/config (committed to git)
git add .dvc/config
git commit -m "add DVC remote"

# Push to remote
dvc push

# Teammate can now pull
dvc pull
```

## Metadata Files

### config.yaml

Auto-generated at training start:

```yaml
run_name: mert95m_baseline
backbone: m-a-p/MERT-v1-95M
num_epochs: 50
learning_rate: 0.0001
weight_decay: 0.01
scheduler: cosine
gradient_clip: 1.0
save_every: 5
eval_every: 1
device: cuda
```

### metadata.yaml

Auto-generated at training completion:

```yaml
run_name: mert95m_baseline
start_time: "2025-01-04T14:30:00Z"
end_time: "2025-01-04T16:45:00Z"
final_metrics:
  best_epoch: 42
  best_val_loss: 0.0234
dataset:
  data_dvc_hash: 8b21e4f497146b0d1a4139220bacd0da.dir
git_commit: 7173f48
device: cuda
```

## Advanced Usage

### Tracking Specific Checkpoints

Instead of tracking entire run, track only best checkpoint:

```bash
dvc add outputs/training/my_run/checkpoints/best.pt
git add outputs/training/my_run/checkpoints/best.pt.dvc
git commit -m "track best checkpoint only"
```

### Checkpoint Tags

Use git tags for important checkpoints:

```bash
# Track experiment
dvc add outputs/training/production_v1
git add outputs/training/production_v1.dvc
git commit -m "track production model v1"

# Tag in git
git tag model-v1.0-production
git push --tags
```

### Experiment Branches

Run experiments in separate git branches:

```bash
# Create experiment branch
git checkout -b experiment/mert330m-ablation

# Train and track
uv run edm train data/annotations --run-name mert330m_ablation
dvc add outputs/training/mert330m_ablation
git add outputs/training/mert330m_ablation.dvc
git commit -m "track mert-330m ablation study"

# Merge if successful
git checkout main
git merge experiment/mert330m-ablation
```

## Troubleshooting

### "File already tracked"

```bash
# Error: outputs/training/my_run is already tracked
# Solution: Remove old .dvc file first
rm outputs/training/my_run.dvc
dvc add outputs/training/my_run
```

### "Cannot find checkpoint"

```bash
# Error: checkpoint file not found
# Solution: Pull from DVC
dvc pull outputs/training/my_run.dvc
```

### "Cache corrupted"

```bash
# Remove local cache
rm -rf .dvc/cache

# Re-pull from remote
dvc pull
```

## Best Practices

1. **Meaningful run names** - Use descriptive names, not timestamps alone
2. **Review before tracking** - Don't track failed/poor experiments
3. **Track baselines** - Always track your baseline for comparison
4. **Document in commits** - Include key metrics in git commit messages
5. **Clean up regularly** - Remove untracked experiments weekly
6. **Use remote storage** - Don't rely on local cache alone

## Example End-to-End Workflow

```bash
# === DAY 1: Initial Training ===

# Train baseline model
uv run edm train data/annotations \
  --run-name mert95m_baseline \
  --epochs 50 \
  --batch-size 4

# Review results
tensorboard --logdir outputs/training/mert95m_baseline/logs
cat outputs/training/mert95m_baseline/metadata.yaml

# Good results! Track with DVC
dvc add outputs/training/mert95m_baseline
git add outputs/training/mert95m_baseline.dvc .gitignore
git commit -m "track mert-95m baseline (val_loss=0.0234, epoch=42)"

# === DAY 2: Experiment with CNN ===

# Try faster CNN backbone
uv run edm train data/annotations \
  --run-name cnn_fast \
  --backbone cnn \
  --epochs 20

# Review
cat outputs/training/cnn_fast/metadata.yaml
# → val_loss=0.0456 (worse than baseline)

# Don't track, just delete
rm -rf outputs/training/cnn_fast/

# === DAY 3: Ablation Study ===

# Try without energy head
uv run edm train data/annotations \
  --run-name ablation_no_energy \
  --no-energy \
  --epochs 50

# Review
cat outputs/training/ablation_no_energy/metadata.yaml
# → val_loss=0.0298 (worse, energy head helps)

# Track for documentation purposes
dvc add outputs/training/ablation_no_energy
git add outputs/training/ablation_no_energy.dvc
git commit -m "track ablation study: no energy head (val_loss=0.0298)"

# === WEEK 2: Setup Remote Storage ===

# Configure S3 remote (one-time setup)
dvc remote add -d s3remote s3://my-edm-experiments/checkpoints
git add .dvc/config
git commit -m "add S3 remote for DVC"

# Push all tracked experiments to S3
dvc push

# === MONTH 2: Teammate Retrieves ===

# Teammate clones repo
git clone https://github.com/user/edm.git
cd edm

# See available experiments
ls outputs/training/*.dvc
# → mert95m_baseline.dvc
# → ablation_no_energy.dvc

# Pull baseline checkpoint
dvc pull outputs/training/mert95m_baseline.dvc

# Load and use
python -c "
import torch
from edm.models.multitask import create_model

# Load checkpoint
ckpt = torch.load('outputs/training/mert95m_baseline/checkpoints/best.pt')

# Recreate model
model = create_model('mert-95m')
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded model from epoch {ckpt[\"epoch\"]}')
"
```

## References

- [DVC Documentation](https://dvc.org/doc)
- [Experiment Tracking Guide](https://dvc.org/doc/use-cases/experiment-tracking)
- [DVC with PyTorch](https://dvc.org/doc/use-cases/model-registry)
