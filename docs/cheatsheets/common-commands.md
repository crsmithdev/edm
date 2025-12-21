# Common Commands

Quick reference for frequently used EDM commands.

## Analysis

```bash
# Analyze single file
uv run edm analyze track.mp3

# Analyze with specific types
uv run edm analyze track.mp3 --types bpm,structure

# Batch analysis
uv run edm analyze *.mp3 --output results.json

# Recursive directory
uv run edm analyze /path/to/tracks/ --recursive
```

## Training

See [Training Cheatsheet](training.md) for complete training reference.

```bash
# Quick commands
just train-quick      # 10 epochs, 2 hours
just train-standard   # 50 epochs, 2-3 hours

# Manual training
uv run edm train data/annotations --audio-dir ~/music --epochs 50
```

## Annotator

```bash
# Start annotator (both servers)
just annotator

# Access: http://localhost:5174
```

## Testing

```bash
# All tests
uv run pytest

# Specific package
uv run pytest packages/edm-lib/tests/

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Just file commands
just test        # All tests
just test-cov    # With coverage
```

## Code Quality

```bash
# Lint
uv run ruff check .                    # Check
uv run ruff check --fix .              # Auto-fix

# Format
uv run ruff format .

# Type check
uv run mypy packages/edm-lib/src/

# All checks
uv run pytest && uv run mypy src/ && uv run ruff check .
```

## Data Management

```bash
# Validate annotations
uv run edm data validate data/annotations

# Import from Rekordbox
uv run edm data import-rekordbox ~/music/rekordbox.xml

# Export to JAMS
uv run edm data export-jams data/annotations --output jams/
```

## Experiment Tracking

```bash
# DVC
dvc add outputs/training/experiment_name
dvc push
dvc pull

# MLflow
mlflow ui                               # Start UI (port 5000)
mlflow experiments list                 # List experiments
mlflow runs list --experiment-name edm-training  # List runs
```

## Development

```bash
# Install dependencies
uv sync

# Run specific package
uv run python -m edm_cli.main analyze track.mp3

# Shell into environment
uv run python
```

## Git Workflow

```bash
# Commit
git add .
git commit -m "description"
git push

# With DVC
dvc add outputs/training/experiment
git add outputs/training/experiment.dvc
git commit -m "track experiment: val_loss=0.0234"
dvc push
git push
```

## Monitoring

```bash
# TensorBoard (training)
tensorboard --logdir outputs/training/

# MLflow (experiments)
mlflow ui

# System resources
nvidia-smi          # GPU
htop                # CPU/RAM
```

## Debugging

```bash
# Verbose logging
uv run edm analyze track.mp3 --log-level DEBUG

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Profile Python code
uv run python -m cProfile -o profile.stats -m edm_cli.main analyze track.mp3
```

## Package Management

```bash
# Install/update
uv sync

# Add dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update all
uv lock --upgrade
uv sync
```

## Common Workflows

### New Feature Development

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Write tests
uv run pytest tests/test_my_feature.py

# 3. Implement
# ... code ...

# 4. Check quality
uv run pytest
uv run mypy src/
uv run ruff check --fix .

# 5. Commit
git add .
git commit -m "add feature: description"
git push origin feature/my-feature
```

### Train and Version Model

```bash
# 1. Train
uv run edm train data/annotations --run-name experiment_1 --epochs 50

# 2. Evaluate in MLflow UI
mlflow ui

# 3. If good, version with DVC
dvc add outputs/training/experiment_1
git add outputs/training/experiment_1.dvc
git commit -m "track experiment_1: val_loss=0.0245"
dvc push
git push
```

### Deploy Annotator

```bash
# 1. Build frontend
cd packages/edm-annotator/frontend
pnpm run build

# 2. Start backend
cd ../backend
gunicorn --bind 0.0.0.0:5000 --workers 4 edm_annotator.app:app

# 3. Serve frontend with Nginx
# See deployment.md for Nginx config
```

## Environment Variables

```bash
# Set common variables
export EDM_AUDIO_DIR=~/music
export EDM_ANNOTATION_DIR=data/annotations
export EDM_LOG_LEVEL=INFO

# Add to ~/.bashrc or ~/.zshrc for persistence
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` |
| `FileNotFoundError: audio` | Check `--audio-dir` or `EDM_AUDIO_DIR` |
| `CUDA out of memory` | Reduce `--batch-size` |
| `pytest: command not found` | Run `uv run pytest` |
| Tests failing | Run `uv sync` and retry |

For detailed troubleshooting, see [Troubleshooting Guide](../reference/troubleshooting.md).

## See Also

- **[Training Cheatsheet](training.md)** - Complete training commands
- **[CLI Reference](../reference/cli.md)** - All CLI options
- **[Documentation Index](../INDEX.md)** - Full documentation map
