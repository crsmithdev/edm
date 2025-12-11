# Training Quick Reference

Fast reference for common model training workflows. For complete documentation, see [training.md](training.md).

## Quick Commands

```bash
# Quick test (10 epochs, small batch)
just train-quick

# Standard training (50 epochs, MERT-95M)
just train-standard

# Full training (100 epochs, MERT-330M)
just train-full

# Resume from checkpoint
just train-resume outputs/training/run_xyz/checkpoints/epoch_20.pt
```

## Common Training Patterns

### 1. First Time Training

```bash
# Prepare data
ls data/annotations/*.yaml  # Verify annotations exist
ls ~/music/*.{mp3,flac,wav}  # Verify audio files exist

# Quick test run (10 epochs, 2 hours)
uv run edm train data/annotations \
    --audio-dir ~/music \
    --epochs 10 \
    --batch-size 4 \
    --output outputs/test_run

# Monitor in TensorBoard
tensorboard --logdir outputs/test_run/logs
```

**Expected results**: Boundary F1 > 0.4 by epoch 10

### 2. Standard Production Training

```bash
# Use training_first_run.yaml config
just train-standard

# Or explicit command
uv run edm train data/annotations \
    --audio-dir ~/music \
    --epochs 50 \
    --batch-size 4 \
    --backbone mert-95m \
    --boundary-head \
    --beat-head \
    --energy-head \
    --output outputs/production

# Runtime: ~2-3 hours on GPU, ~20-30 hours on CPU
```

**Expected results**: Boundary F1 > 0.7 by epoch 50

### 3. Fast Iteration (CNN Backbone)

```bash
# Train simple CNN for quick experiments
uv run edm train data/annotations \
    --audio-dir ~/music \
    --backbone cnn \
    --epochs 20 \
    --batch-size 16 \
    --output outputs/cnn_fast

# Runtime: ~30 minutes on GPU
```

### 4. Resume Training

```bash
# Find checkpoint
ls outputs/training/*/checkpoints/

# Resume from epoch
uv run edm train data/annotations \
    --resume outputs/training/run_xyz/checkpoints/epoch_20.pt \
    --epochs 100  # Will train from epoch 21 to 100
```

## Config File Training

```bash
# Use predefined config
uv run edm train --config configs/training_first_run.yaml

# Override specific parameters
uv run edm train --config configs/training_first_run.yaml \
    --epochs 100 \
    --batch-size 8
```

## Monitoring Training

### TensorBoard

```bash
# Single run
tensorboard --logdir outputs/training/run_xyz/logs

# Compare all runs
tensorboard --logdir outputs/training/

# Access at http://localhost:6006
```

**Key metrics to watch**:
- `val/boundary_f1` - Main metric (target: > 0.7)
- `val/total_loss` - Should decrease steadily
- `train/total_loss` vs `val/total_loss` - Check for overfitting

### Live Output

Training prints progress every 10 batches:

```
Epoch 5/50 [=========>            ] 45% | Batch 120/267 | Loss: 0.234 | F1: 0.653
```

## Troubleshooting Checklist

### Training Not Starting

```bash
# Verify annotations
ls data/annotations/*.yaml | wc -l  # Should show count

# Verify audio files exist
python -c "
import yaml
from pathlib import Path
ann = yaml.safe_load(Path('data/annotations/track1.yaml').read_text())
print(f\"Audio: {ann['audio']['file']}\")
print(f\"Exists: {Path(ann['audio']['file']).exists()}\")
"

# Test dataset loading
uv run python -c "
from edm.training.dataset import EDMDataset
from pathlib import Path
dataset = EDMDataset(
    annotation_dir=Path('data/annotations'),
    audio_dir=Path('~/music').expanduser(),
)
print(f'Dataset size: {len(dataset)}')
"
```

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 2

# Reduce audio duration
--duration 15.0

# Use smaller backbone
--backbone cnn

# Disable some task heads
--no-energy --no-beat
```

### Poor Performance

| Symptom | Solution |
|---------|----------|
| F1 stuck at 0 | Check annotation quality, increase `--boundary-tolerance` |
| Train loss decreasing, val loss increasing | Overfitting: add more data, reduce epochs |
| Both losses high | Underfitting: increase epochs, larger model |
| Training very slow | Use GPU, increase `--batch-size`, reduce `--workers` |

### Loss is NaN

```bash
# Reduce learning rate
--lr 1e-5

# Check for bad annotations (missing/invalid data)
uv run edm data validate data/annotations

# Disable problematic task heads
--no-label  # Label classification can be unstable
```

## Hardware Requirements

| Configuration | GPU | VRAM | Training Time (50 epochs, 100 tracks) |
|---------------|-----|------|---------------------------------------|
| Quick (CNN) | Optional | 2GB | ~30-45 min (GPU), ~5-8 hrs (CPU) |
| Standard (MERT-95M) | Recommended | 4GB+ | ~2-3 hours |
| Full (MERT-330M) | Required | 8GB+ | ~5-6 hours |

**CPU-only training**: Use `--backbone cnn` (10x slower but works)

## Data Requirements

| Dataset Size | Quality | Expected F1 |
|--------------|---------|-------------|
| 20-50 tracks | Mixed | 0.4-0.5 |
| 50-100 tracks | Good | 0.6-0.7 |
| 100+ tracks | High quality | 0.7-0.8+ |

**Minimum**: 50 annotated tracks
**Recommended**: 100+ tracks with diverse EDM subgenres

## File Locations Reference

| Item | Location |
|------|----------|
| Annotations | `data/annotations/*.yaml` |
| Audio files | `~/music` (default) or specify with `--audio-dir` |
| Training outputs | `outputs/training/{run_name}/` |
| Checkpoints | `outputs/training/{run_name}/checkpoints/` |
| TensorBoard logs | `outputs/training/{run_name}/logs/` |
| Training configs | `configs/*.yaml` |
| Model code | `packages/edm-lib/src/edm/models/` |
| Training code | `packages/edm-lib/src/edm/training/` |

## Common Workflows

### Experiment Iteration

```bash
# 1. Quick test (10 epochs)
just train-quick

# 2. Review results
tensorboard --logdir outputs/test_run/logs
cat outputs/test_run/metadata.yaml

# 3. If promising, run full training
just train-standard

# 4. Compare runs in TensorBoard
tensorboard --logdir outputs/training/
```

### Hyperparameter Tuning

```bash
# Baseline
uv run edm train data/annotations --epochs 50 --run-name baseline

# Experiment 1: Larger model
uv run edm train data/annotations --backbone mert-330m --epochs 50 --run-name exp_large

# Experiment 2: More data per sample
uv run edm train data/annotations --duration 60.0 --epochs 50 --run-name exp_long

# Experiment 3: Focus on boundaries
uv run edm train data/annotations \
    --boundary-weight 2.0 \
    --no-energy --no-beat \
    --epochs 50 \
    --run-name exp_boundary_only

# Compare in TensorBoard
tensorboard --logdir outputs/training/
```

### Production Model Training

```bash
# 1. Train with best hyperparameters
uv run edm train data/annotations \
    --audio-dir ~/music \
    --backbone mert-95m \
    --epochs 100 \
    --batch-size 8 \
    --run-name production_v1 \
    --experiment-name edm-production

# 2. Evaluate on test set
# TODO: Implement test set evaluation

# 3. Version with DVC
dvc add outputs/training/production_v1
git add outputs/training/production_v1.dvc
git commit -m "add production model v1 (val_f1=0.78)"

# 4. Deploy
# TODO: Model deployment workflow
```

## Next Steps

- **After training**: See [training.md](training.md#model-evaluation) for evaluation
- **Debugging issues**: See [training.md](training.md#troubleshooting) for detailed troubleshooting
- **Architecture details**: See [training.md](training.md#architecture) for model architecture
- **Advanced tuning**: See [training.md](training.md#advanced-usage) for advanced options

## Getting Help

1. Check [training.md](training.md) for detailed documentation
2. Check [docs/agent-guide.md](agent-guide.md) for code navigation
3. Review training config: `configs/training_first_run.yaml`
4. Inspect model code: `packages/edm-lib/src/edm/models/`
5. Check training implementation: `packages/edm-lib/src/edm/training/trainer.py`
