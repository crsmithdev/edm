# EDM Development Notes

Quick reference for common development tasks.

## Model Training

### Quick Commands

```bash
# Quick test (10 epochs, ~2 hours)
just train-quick

# Standard training (50 epochs, ~3-4 hours)
just train-standard

# Full training (100 epochs, large model)
just train-full

# Using config file
just train-config configs/training_first_run.yaml

# Resume from checkpoint
just train-resume outputs/training/run_xyz/checkpoints/epoch_20.pt
```

### First Time Training

```bash
# 1. Verify data
ls data/annotations/*.yaml | wc -l  # Should show annotation count
ls ~/music/*.{mp3,flac,wav} | head -5  # Verify audio files

# 2. Quick test run
just train-quick

# 3. Monitor in TensorBoard
tensorboard --logdir outputs/test_run/logs
# Open http://localhost:6006

# 4. Check results
cat outputs/test_run/metadata.yaml
ls outputs/test_run/checkpoints/
```

### Common Issues

- **OOM errors**: Reduce batch size with `--batch-size 2` or use `--backbone cnn`
- **No data found**: Check paths in YAML annotations match `--audio-dir`
- **F1 stuck at 0**: Check annotation quality, increase boundary tolerance
- **Slow training**: Ensure GPU is being used (`nvidia-smi`), reduce `--workers`

See [docs/training-quickref.md](../docs/training-quickref.md) for detailed troubleshooting.

## Running the Annotator Web App

The EDM annotator is a React + Flask application for annotating track structures.

### Quick Start

```bash
# From repo root
just annotator
```

This starts:
- Backend API: http://localhost:5000 (Flask)
- Frontend: http://localhost:5173 (Vite + React + TypeScript)

### First Time Setup

```bash
# Install backend (from repo root)
uv sync

# Install frontend
cd packages/edm-annotator/frontend
npm install
```

### Manual Start (Alternative)

If you need to run servers separately:

```bash
# Terminal 1: Backend
cd packages/edm-annotator
uv run edm-annotator --env development --port 5000

# Terminal 2: Frontend
cd packages/edm-annotator/frontend
npm run dev
```

## Package Structure

- **packages/edm-annotator/** - Annotator package root
  - **pyproject.toml** - Python package config (v2.0.0)
  - **backend/src/edm_annotator/** - Flask API
  - **frontend/src/** - React app (TypeScript + Vite)
  - **run-dev.sh** - Dev server launcher script

## Common Issues

### Backend won't start
- Run `uv sync` from repo root
- Verify with: `uv run edm-annotator --help`

### Frontend won't start
- Run `npm install` in packages/edm-annotator/frontend
- Check Node version: `node --version` (needs 18+)

### Port already in use
- Backend: Change port with `--port 5001`
- Frontend: Vite will auto-increment (5174, 5175, etc.)

## Documentation

- **packages/edm-annotator/QUICKSTART.md** - Detailed setup guide
- **packages/edm-annotator/README.md** - Full documentation
- **justfile** - All available commands (run `just` to list)
