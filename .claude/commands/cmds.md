---
description: Show project commands
---

# EDM Project Commands

## Development
```bash
just test          # Run all tests
just lint          # Lint and format code
just typecheck     # Type check with mypy
just install       # Install dependencies
just clean         # Remove caches
just check         # Run all checks in parallel
```

## CLI Usage
```bash
uv run edm analyze <audio_file>              # Analyze single track
uv run edm train --config <config.yaml>      # Train model
uv run edm evaluate <model_path> <test_dir>  # Evaluate model
```

## Training
```bash
just train         # Train with default config (configs/training_first_run.yaml)
uv run edm train --config configs/custom.yaml --epochs 50 --batch-size 8
```

## Project Structure
- `src/cli/` - CLI commands (analyze, train, evaluate)
- `src/edm/` - Core library (analysis, training, models)
- `data/` - Annotations (YAML)
- `configs/` - Training configs
- `experiments/` - Training runs
