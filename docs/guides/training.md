# Model Training

Train machine learning models for EDM structure detection using annotated data.

## Quick Start

```bash
# Train with default settings (30-second segments, MERT-95M backbone)
uv run edm train data/annotations --audio-dir ~/music

# Train for 10 epochs with custom batch size
uv run edm train data/annotations --audio-dir ~/music --epochs 10 --batch-size 8

# Train with CNN backbone (faster, less accurate)
uv run edm train data/annotations --backbone cnn --epochs 20

# Train only boundary detection (disable other tasks)
uv run edm train data/annotations --no-energy --no-beat --no-label

# Resume from checkpoint
uv run edm train data/annotations --resume outputs/training/epoch_20.pt
```

## Architecture

The training pipeline uses a multi-task learning approach:

### Backbone Options

1. **MERT-95M** (default)
   - Pretrained music transformer (95M parameters)
   - Best accuracy
   - ~5-10s per track on GPU
   - Requires: GPU with 4GB+ VRAM

2. **MERT-330M**
   - Larger pretrained transformer (330M parameters)
   - Highest accuracy
   - ~15-20s per track on GPU
   - Requires: GPU with 8GB+ VRAM

3. **SimpleCNN**
   - Lightweight CNN backbone
   - Fastest training
   - Lower accuracy
   - Works on CPU

### Task Heads

Multi-task learning with four prediction heads:

1. **Boundary Detection** (priority 1)
   - Frame-level binary classification
   - Detects structure boundaries (intro→buildup, buildup→drop, etc.)
   - Loss: Focal BCE (handles class imbalance)

2. **Energy Prediction** (priority 2)
   - Frame-level regression (3 bands: bass/mid/high)
   - Predicts energy levels for drops, buildups, breakdowns
   - Loss: MSE with high-energy weighting

3. **Beat Detection** (priority 3)
   - Frame-level binary classification
   - Improves beat tracking beyond metadata
   - Loss: Focal BCE

4. **Label Classification** (priority 4, optional)
   - Frame-level multi-class (5 classes: intro/buildup/drop/breakdown/outro)
   - Disabled by default (less stable)
   - Loss: Cross-entropy

## Training Data

The training pipeline expects:

### Directory Structure

```
data/
├── annotations/     # YAML annotation files (required)
│   ├── track1.yaml
│   ├── track2.yaml
│   └── ...
└── jams/           # JAMS format (optional)
```

### Audio Files

Audio files can be:
- In annotation paths (e.g., `/home/user/music/track.flac`)
- In `--audio-dir` (overrides annotation paths)
- Must be readable by librosa (FLAC, MP3, WAV, etc.)

### Annotation Format

YAML files with structure boundaries:

```yaml
metadata:
  tier: 1
  confidence: 0.95
audio:
  file: /path/to/track.flac
  duration: 240.0
  bpm: 128.0
  downbeat: 0.0
  time_signature: [4, 4]
structure:
  - bar: 1
    label: intro
    time: 0.0
    confidence: 1.0
  - bar: 17
    label: buildup
    time: 30.0
    confidence: 0.9
  - bar: 25
    label: drop
    time: 45.0
    confidence: 1.0
```

Valid labels: `intro`, `buildup`, `drop`, `breakdown`, `outro`

## Training Configuration

### Hyperparameters

```bash
--epochs 50              # Number of training epochs
--batch-size 4          # Batch size (decrease if OOM)
--lr 0.0001             # Learning rate (1e-4)
--train-split 0.8       # 80% train, 20% validation
--duration 30.0         # Audio segment length (seconds)
--workers 4             # Dataloader workers (parallel loading)
```

### Loss Weights

Balance multi-task learning:

```bash
--boundary-weight 1.0   # Boundary detection (highest priority)
--energy-weight 0.5     # Energy prediction
--beat-weight 0.5       # Beat detection
--label-weight 0.3      # Label classification (lowest priority)
```

### Task Selection

Enable/disable specific tasks:

```bash
--boundary / --no-boundary
--energy / --no-energy
--beat / --no-beat
--label / --no-label
```

## Training Output

### Directory Structure

Training outputs are organized by run:

```
outputs/training/
├── run_20250104_143052_mert95m_default/
│   ├── checkpoints/
│   │   ├── best.pt           # Best model checkpoint
│   │   ├── epoch_5.pt         # Periodic checkpoints
│   │   └── epoch_10.pt
│   ├── logs/                  # TensorBoard logs
│   │   └── events.out.tfevents.*
│   ├── config.yaml            # Training configuration
│   └── metadata.yaml          # Run metadata
└── run_20250105_083012_cnn_fast/
    └── ...
```

### Run Naming

Runs are automatically named: `run_{YYYYMMDD}_{HHMMSS}_{backbone}_{description}`

```bash
# Auto-generated name (default description)
uv run edm train data/annotations
# → outputs/training/run_20250104_143052_mert95m_default/

# Custom run name
uv run edm train data/annotations --run-name baseline_v1
# → outputs/training/baseline_v1/
```

### Checkpoints

Checkpoint contents:
- Model state dict
- Optimizer state
- Scheduler state
- Training epoch
- Best validation loss
- Configuration

### TensorBoard

Monitor training:

```bash
# Single run
tensorboard --logdir outputs/training/run_20250104_143052_mert95m_default/logs

# Compare multiple runs
tensorboard --logdir outputs/training/
```

Metrics logged:
- `train/total_loss`, `train/boundary_loss`, `train/energy_loss`, etc.
- `val/total_loss`, `val/boundary_loss`, etc.
- `train/learning_rate`

## Experiment Tracking with DVC

Training outputs are git-ignored but can be selectively versioned with DVC for experiment tracking.

### Workflow

```bash
# 1. Train a model
uv run edm train data/annotations --run-name mert95m_baseline

# 2. Review results
tensorboard --logdir outputs/training/mert95m_baseline/logs
cat outputs/training/mert95m_baseline/metadata.yaml

# 3. Version with DVC (if worth keeping)
dvc add outputs/training/mert95m_baseline
git add outputs/training/mert95m_baseline.dvc .gitignore
git commit -m "track mert-95m baseline (val_loss=0.0234)"

# 4. Retrieve later
dvc pull outputs/training/mert95m_baseline.dvc
```

### Remote Storage (Future)

Configure remote storage for team collaboration:

```bash
# S3
dvc remote add -d s3remote s3://my-bucket/edm-checkpoints
dvc push

# Google Cloud Storage
dvc remote add -d gcs gs://my-bucket/edm-checkpoints
dvc push

# NAS/Network Drive
dvc remote add -d nas /mnt/shared/edm-checkpoints
dvc push
```

See [Model Management Guide](model-management.md) for comprehensive DVC and MLflow workflow guide.

## Model Evaluation

After training, evaluate the model:

```bash
# TODO: Implement inference and evaluation with trained model
# This will integrate with existing edm evaluate command
```

## Best Practices

### Data

- **Minimum 50 annotated tracks** for meaningful training
- **100+ tracks recommended** for good generalization
- **Diverse genres** within EDM (house, dubstep, trance, etc.)
- **High-quality annotations** (tier 1-2) for best results

### Hardware

- **GPU strongly recommended** for MERT backbones
- **4GB+ VRAM** for batch_size=4 with MERT-95M
- **8GB+ VRAM** for batch_size=8 or MERT-330M
- **CPU only**: Use `--backbone cnn` (10x slower but works)

### Training Time

Estimates for 100 tracks, 50 epochs:

- MERT-95M on GPU: ~2-3 hours
- MERT-330M on GPU: ~5-6 hours
- SimpleCNN on GPU: ~30-45 minutes
- SimpleCNN on CPU: ~5-8 hours

### Hyperparameter Tuning

Start with defaults, then adjust:

1. **Overfitting** (train loss << val loss):
   - Increase `--duration` (more data per sample)
   - Add more training data
   - Decrease model size (`--backbone cnn`)

2. **Underfitting** (high train and val loss):
   - Increase `--epochs` (more training)
   - Increase model size (`--backbone mert-330m`)
   - Decrease `--duration` (easier task)

3. **One task dominates**:
   - Adjust loss weights (`--boundary-weight`, etc.)
   - Disable low-priority tasks (`--no-label`)

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 2

# Reduce segment duration
--duration 15.0

# Use smaller backbone
--backbone cnn

# Reduce workers
--workers 2
```

### Slow Training

```bash
# Use GPU (check with nvidia-smi)
# Increase batch size (if memory allows)
--batch-size 8

# Increase workers
--workers 8

# Reduce validation frequency
--eval-every 5
```

### Poor Convergence

```bash
# Adjust learning rate
--lr 0.0001  # Default
--lr 0.00005  # Lower for fine-tuning
--lr 0.0003   # Higher for CNN from scratch

# Increase epochs
--epochs 100

# Focus on one task
--no-energy --no-beat --no-label
```

## Advanced Usage

### Custom Loss Weights

Prioritize boundary detection:

```bash
uv run edm train data/annotations \
  --boundary-weight 2.0 \
  --energy-weight 0.3 \
  --beat-weight 0.3 \
  --label-weight 0.1
```

### Training from Scratch (CNN)

```bash
uv run edm train data/annotations \
  --backbone cnn \
  --epochs 100 \
  --lr 0.001 \
  --batch-size 16
```

### Fine-tuning MERT

```bash
# Phase 1: Train heads only (frozen backbone)
# TODO: Add --freeze-backbone flag

# Phase 2: Fine-tune full model
uv run edm train data/annotations \
  --backbone mert-95m \
  --resume outputs/training/phase1_best.pt \
  --epochs 50 \
  --lr 0.00005
```

## Implementation Details

### Data Pipeline

1. **Dataset**: `edm.training.dataset.EDMDataset`
   - Loads YAML annotations
   - Loads audio with librosa
   - Creates frame-level targets (boundary, energy, beat, label)
   - Handles variable-length sequences

2. **DataLoader**: PyTorch DataLoader with custom collation
   - Pads sequences to max length in batch
   - Parallel loading with `num_workers`
   - Pin memory for GPU transfer

### Training Loop

1. **Optimizer**: AdamW with weight decay
2. **Scheduler**: Cosine annealing (default)
3. **Gradient clipping**: Max norm = 1.0
4. **Mixed precision**: TODO (would speed up MERT)

### Model Architecture

See `src/edm/models/`:
- `backbone.py`: MERTBackbone, SimpleCNNBackbone
- `heads.py`: BoundaryHead, EnergyHead, BeatHead, LabelHead
- `multitask.py`: MultiTaskModel combining backbone + heads

### Loss Functions

See `src/edm/training/losses.py`:
- `MultiTaskLoss`: Weighted combination
- `BoundaryF1Loss`: F1-optimized (alternative)
- `WeightedMSELoss`: Energy-weighted MSE (alternative)

## References

- MERT: [Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-95M)
- Multi-task Learning: [An Overview](https://arxiv.org/abs/1706.05098)
- Focal Loss: [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002)
