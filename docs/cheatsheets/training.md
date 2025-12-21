# Training Quick Reference

Essential commands for model training. For complete documentation, see [Training Guide](../guides/training.md).

## Common Commands

```bash
# First-time training (test run)
uv run edm train data/annotations --audio-dir ~/music --epochs 10 --batch-size 4

# Standard production training
uv run edm train data/annotations --audio-dir ~/music --epochs 50 --backbone mert-95m

# Fast iteration with CNN
uv run edm train data/annotations --audio-dir ~/music --backbone cnn --epochs 20

# Resume from checkpoint
uv run edm train data/annotations --resume outputs/training/run_xyz/checkpoints/epoch_20.pt
```

## Quick Justfile Commands

```bash
just train-quick      # 10 epochs, small batch (2 hours)
just train-standard   # 50 epochs, MERT-95M (2-3 hours)
just train-full       # 100 epochs, MERT-330M (5-6 hours)
```

## Monitoring

```bash
# TensorBoard
tensorboard --logdir outputs/training/

# MLflow UI
mlflow ui
```

## Common Configurations

```bash
# Custom hyperparameters
--epochs 50           # Training epochs
--batch-size 4        # Batch size (reduce if OOM)
--lr 0.0001          # Learning rate
--backbone mert-95m  # Model backbone (mert-95m, mert-330m, cnn)

# Task heads
--boundary-head      # Segment boundaries
--beat-head          # Beat detection
--energy-head        # Energy levels

# Hardware
--device cuda        # Use GPU (cuda or cpu)
--workers 8          # Data loading workers
```

## Quick Troubleshooting

```bash
# Out of memory
--batch-size 2 --duration 15.0

# Training is NaN
--lr 1e-5

# Slow training
--workers 8 --device cuda

# Model not improving
--epochs 100 --lr 0.00005
```

For detailed troubleshooting, see:
- [Training Guide - Troubleshooting](../guides/training.md#troubleshooting)
- [Troubleshooting Guide - Training](../reference/troubleshooting.md#training)

## Expected Results

| Configuration | Boundary F1 | Time (GPU) |
|--------------|-------------|------------|
| Quick (10 epochs) | > 0.4 | ~2 hours |
| Standard (50 epochs) | > 0.7 | ~2-3 hours |
| Full (100 epochs) | > 0.8 | ~5-6 hours |

## See Also

- **[Training Guide](../guides/training.md)** - Complete training documentation
- **[Model Management](../guides/model-management.md)** - Experiment tracking and versioning
- **[CLI Reference](../reference/cli.md)** - All CLI commands
