# Common Tasks Quick Reference

Fast reference for frequently needed operations. Links to detailed documentation.

## Package Management

```bash
# Install all dependencies
uv sync

# Add new dependency to edm-lib
cd packages/edm-lib
uv add package-name

# Add dev dependency
uv add --dev pytest-mock

# Update lockfile
uv lock
```

## Running Code

### Analysis

```bash
# Analyze single file
uv run edm analyze track.mp3

# Analyze multiple files with parallelism
uv run edm analyze *.mp3 --workers 8

# Save to JSON
uv run edm analyze *.mp3 --output results.json

# Via justfile
just analyze track.mp3 --types bpm,structure
```

### Training

```bash
# Quick test (10 epochs)
just train-quick

# Standard production (50 epochs)
just train-standard

# Full training (100 epochs, large model)
just train-full

# Using config file
just train-config configs/training_first_run.yaml

# Resume training
just train-resume outputs/training/run_xyz/checkpoints/epoch_20.pt

# Monitor training
tensorboard --logdir outputs/training/
```

See [training-quickref.md](training-quickref.md) for detailed training workflows.

### Annotator

```bash
# Start full stack (backend + frontend)
just annotator

# Access at:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:5000/api
```

**Environment setup:**
```bash
export EDM_AUDIO_DIR=/path/to/music
export EDM_ANNOTATION_DIR=/path/to/annotations
```

**Quality checks (run from project root):**
```bash
# All checks (backend + frontend in parallel)
just check

# Backend only
just lint    # Ruff linting with auto-fix
just types   # Mypy type checking
just fmt     # Ruff formatting

# Frontend only
cd packages/edm-annotator/frontend
npx tsc --noEmit      # Type check
npx eslint src/       # Lint
pnpm test            # Tests (when implemented)
```

**Development:**
```bash
# Install frontend dependencies
cd packages/edm-annotator/frontend
pnpm install

# Build frontend for production
pnpm build

# Backend tests
cd packages/edm-annotator/backend
pytest
```

**Features:**
- Dual waveform display (overview + detail with scrolling)
- Drag-to-scrub playback control
- Cue point system (C/R keys)
- Quantize snapping with Shift bypass
- Boundary marking (Ctrl+click or B key)
- Region labeling (intro/buildup/main/breakdown/outro)
- Complete keyboard shortcuts (? for help)

## Testing

```bash
# All tests
just test

# Specific package
uv run pytest packages/edm-lib/tests/

# With coverage
just test-cov

# Specific test file
uv run pytest packages/edm-lib/tests/unit/test_bpm.py

# Specific test function
uv run pytest packages/edm-lib/tests/unit/test_bpm.py::test_analyze_bpm

# Run tests in parallel
uv run pytest -n auto
```

## Code Quality

```bash
# Run all checks (lint, format, type check, tests)
just check

# Individual checks
just fmt      # Format code
just lint     # Lint with auto-fix
just types    # Type check

# Manual commands
uv run ruff format .
uv run ruff check --fix .
uv run mypy packages/edm-lib/src/
```

## Finding Code

### Search for code patterns

```bash
# Find function definitions
grep -r "def analyze_bpm" packages/

# Find imports
grep -r "from edm.analysis import" packages/

# Find class definitions
grep -r "class.*Trainer" packages/

# Find config usage
grep -r "TrainingConfig" packages/
```

### Key file locations

| Feature | Location |
|---------|----------|
| BPM detection | `packages/edm-lib/src/edm/analysis/bpm.py` |
| Structure detection | `packages/edm-lib/src/edm/analysis/structure.py` |
| Model training | `packages/edm-lib/src/edm/training/trainer.py` |
| Model architectures | `packages/edm-lib/src/edm/models/` |
| CLI commands | `packages/edm-cli/src/edm_cli/commands/` |
| Annotations | `data/annotations/*.yaml` |
| Training configs | `configs/*.yaml` |
| Documentation | `docs/*.md` |

## Debugging

### Check training data

```bash
# Count annotations
ls data/annotations/*.yaml | wc -l

# Validate annotations
uv run edm data validate data/annotations

# Check audio paths in annotations
python -c "
import yaml
from pathlib import Path
for f in Path('data/annotations').glob('*.yaml'):
    ann = yaml.safe_load(f.read_text())
    audio = Path(ann['audio']['file'])
    if not audio.exists():
        print(f'Missing: {audio}')
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
print(f'First sample: {dataset[0]}')
"
```

### Check GPU availability

```bash
# NVIDIA GPU
nvidia-smi

# In Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Monitor training

```bash
# TensorBoard (single run)
tensorboard --logdir outputs/training/run_xyz/logs

# TensorBoard (compare all runs)
tensorboard --logdir outputs/training/

# Check training outputs
ls outputs/training/*/checkpoints/
cat outputs/training/*/metadata.yaml

# Live training log
tail -f outputs/training/run_xyz/train.log
```

## Data Management

### Annotations

```bash
# Import Rekordbox XML
uv run edm data import-rekordbox ~/music/rekordbox.xml

# Export to JAMS format
uv run edm data export-jams data/annotations --output jams/

# Validate annotations
uv run edm data validate data/annotations
```

### Audio Files

```bash
# Find audio files
find ~/music -name "*.flac" -o -name "*.mp3" | wc -l

# Check audio file metadata
uv run python -c "
import mutagen
from pathlib import Path
audio = mutagen.File(Path('~/music/track.flac').expanduser())
print(f'Duration: {audio.info.length}s')
print(f'Sample rate: {audio.info.sample_rate}Hz')
print(f'BPM: {audio.get(\"bpm\", [\"N/A\"])[0]}')
"
```

## Git Workflows

```bash
# Check status
just ship  # or git status

# Create feature branch
git checkout -b feature/new-detector

# Commit changes (with Claude commit message)
git add .
git commit  # Will invoke pre-commit hooks

# Push and create PR
git push -u origin feature/new-detector
gh pr create --title "Add new detector" --body "Description..."
```

## Documentation

### Finding docs

| Topic | File |
|-------|------|
| Training workflows | [training-quickref.md](training-quickref.md) |
| Complete training guide | [training.md](training.md) |
| Architecture overview | [architecture.md](architecture.md) |
| Project structure | [project-structure.md](project-structure.md) |
| CLI reference | [cli-reference.md](cli-reference.md) |
| Development setup | [development.md](development.md) |
| Testing | [testing.md](testing.md) |
| Agent navigation | [agent-guide.md](agent-guide.md) |

### Building documentation

```bash
# View in browser (if using mkdocs or similar)
# Currently: docs are markdown files, read directly

# Check for broken links
grep -r "\[.*\](.*\.md)" docs/ | grep -v "^docs/.*:.*docs/"
```

## Performance Profiling

### Training performance

```bash
# Profile training step
uv run python -m cProfile -o profile.stats \
    -m edm_cli.commands.train data/annotations --epochs 1

# View profile
uv run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

### Analysis performance

```bash
# Time analysis
time uv run edm analyze track.mp3

# Benchmark multiple workers
for w in 1 2 4 8; do
    echo "Workers: $w"
    time uv run edm analyze *.mp3 --workers $w
done
```

## Package-Specific Commands

### edm-lib

```bash
# Run lib tests
uv run pytest packages/edm-lib/tests/

# Type check lib
uv run mypy packages/edm-lib/src/

# Install lib in editable mode
uv pip install -e packages/edm-lib
```

### edm-cli

```bash
# Run CLI tests
uv run pytest packages/edm-cli/tests/

# Test CLI commands
uv run edm --help
uv run edm analyze --help
uv run edm train --help
```

### edm-annotator

```bash
# Install frontend dependencies
cd packages/edm-annotator/frontend
npm install

# Run frontend tests
npm test

# Run frontend type check
npm run type-check

# Build frontend for production
npm run build
```

## Troubleshooting

### Common errors and solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'edm'` | Run `uv sync` from repo root |
| `FileNotFoundError: audio file not found` | Check `--audio-dir` matches annotation paths |
| `CUDA out of memory` | Reduce `--batch-size` or `--duration` |
| `ImportError: No module named 'torch'` | Run `uv sync` (torch is in dependencies) |
| `pytest: command not found` | Run `uv run pytest` instead |
| Training loss is NaN | Reduce learning rate `--lr 1e-5` |

### Getting help

1. Check documentation in `docs/`
2. Search codebase: `grep -r "pattern" packages/`
3. Check [agent-guide.md](agent-guide.md) for code locations
4. Read test files for usage examples
5. Check justfile for available commands: `just --list`
