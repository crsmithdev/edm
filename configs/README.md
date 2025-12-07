# Training Configurations

## Available Configs

### `training_first_run.yaml`
**Purpose**: First real training run for validating the pipeline
**Hardware**: RTX 5070 (12GB VRAM)
**Data**: 356 annotations, min_confidence=0.8
**Runtime**: 2-3 hours (30 epochs)
**Strategy**: Conservative - frozen backbone, 30s clips, batch_size=8

**Use when**: Initial pipeline validation, baseline metrics

### `dataset_config.yaml`
**Purpose**: JAMS-based dataset filtering and splits
**Format**: JAMS standard annotations
**Filtering**: Tier 2-3, confidence ≥0.7, rekordbox hot cues only

## Quick Start

```bash
# 1. Validate data
uv run edm data validate data/annotations/*.yaml --min-confidence 0.8

# 2. Check dataset stats
uv run edm data stats data/annotations/ --min-confidence 0.8

# 3. Run training
uv run edm train \
  --config configs/training_first_run.yaml \
  --output experiments/first_run \
  --verbose

# 4. Monitor (separate terminal)
tensorboard --logdir experiments/first_run/runs
```

## Expected Results (first_run)

| Metric | Epoch 10 | Epoch 30 | Notes |
|--------|----------|----------|-------|
| val_boundary_f1 | 0.4-0.5 | 0.6-0.7 | Main metric |
| val_beat_f1 | 0.5-0.6 | 0.7-0.8 | If beat grids available |
| val_label_acc | 0.5-0.6 | 0.7-0.8 | Section labels |
| train_loss | ~0.8 | ~0.3 | Should decrease smoothly |
| val_loss | ~0.9 | ~0.4 | Should track train_loss |

## Next Steps After First Run

### If Training Succeeds
1. **Analyze predictions**: `edm analyze --detector ml --validate`
2. **Check alignment**: Cross-validate against beat grids (XVAL)
3. **Tune hyperparameters**: Try configs below

### If Overfitting (val_loss diverges from train_loss)
- Increase dropout, weight_decay
- Enable augmentation
- Reduce model capacity

### If Underfitting (both losses plateau high)
- Unfreeze backbone earlier
- Increase model capacity (mert-330m)
- More epochs
- Higher learning rate

## Config Templates

### Full Track Training
```yaml
# Copy training_first_run.yaml, modify:
data:
  duration: null  # Full tracks instead of 30s clips
  batch_size: 2   # Reduce batch size for memory
training:
  num_epochs: 50  # Longer training needed
```

### With Data Augmentation
```yaml
# Add to training_first_run.yaml:
data:
  augment: true
augmentation:
  time_stretch: [0.95, 1.05]  # ±5% tempo
  pitch_shift: [-2, 2]        # ±2 semitones
  noise_injection: 0.001      # Light noise
```

### High Confidence Only
```yaml
# Modify dataset config:
data:
  tier_filter: 2             # Only Tier 2 (auto-cleaned)
  min_confidence: 0.9        # Very high threshold
```

## MLflow Experiments

Track different experiments:

```yaml
mlflow:
  experiment_name: "edm-structure-baseline"
  run_name: "mert95m_frozen_30ep"  # Descriptive name
```

View results:
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## Hardware Optimization

### RTX 5070 (12GB VRAM)
- batch_size: 8 (30s clips) or 2 (full tracks)
- backbone: mert-95m or mert-330m
- mixed_precision: true

### RTX 3060 (6GB VRAM)
- batch_size: 4 (30s clips)
- backbone: mert-95m only
- gradient_checkpointing: true

### CPU Only
- batch_size: 2
- backbone: cnn (not MERT)
- num_workers: 0
