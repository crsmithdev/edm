# Troubleshooting Guide

Common issues and solutions for the EDM project.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [CLI and Analysis](#cli-and-analysis)
- [Training](#training)
- [Annotator Application](#annotator-application)
- [Experiment Tracking](#experiment-tracking)
- [Performance Issues](#performance-issues)
- [Claude Code Integration](#claude-code-integration)
- [Getting Help](#getting-help)

---

## Installation and Setup

### `ModuleNotFoundError: No module named 'edm'`

**Solution**: Run `uv sync` from repository root

```bash
cd /path/to/edm
uv sync
```

### `ImportError: No module named 'torch'`

**Solution**: PyTorch is in dependencies, reinstall:

```bash
uv sync
```

### `pytest: command not found`

**Solution**: Run pytest through uv:

```bash
uv run pytest
```

### Python version issues

**Check version**:

```bash
python --version  # Needs 3.12+
```

**Solution**: Install Python 3.12+ or use uv to manage Python versions:

```bash
uv python install 3.12
uv python pin 3.12
```

### Node.js/Frontend issues

**Check version**:

```bash
node --version  # Needs 18+
```

**Reinstall dependencies**:

```bash
cd packages/edm-annotator/frontend
rm -rf node_modules .vite
pnpm install
```

---

## CLI and Analysis

### `FileNotFoundError: audio file not found`

**Cause**: Audio file path in annotation doesn't match actual file location

**Solution**: Check that `--audio-dir` matches the path prefix in annotations:

```bash
# If annotation has: audio: { file: "music/track.mp3" }
# Then use:
edm analyze annotations/track.yaml --audio-dir music/
```

### BPM detection fails or gives wrong results

**Symptoms**: BPM is half/double expected value, or completely wrong

**Solutions**:

1. **Check audio file**: Ensure file is valid and not corrupted
   ```bash
   # Test with media player first
   ffplay audio/track.mp3
   ```

2. **Adjust BPM range**: Use `--min-bpm` and `--max-bpm`
   ```bash
   edm analyze track.mp3 --min-bpm 120 --max-bpm 150
   ```

3. **Force specific tempo**: Use `--tempo` to override detection
   ```bash
   edm analyze track.mp3 --tempo 128
   ```

### Structure detection produces poor results

**Solutions**:

1. **Check reference annotations**: Ensure you have good training data
   ```bash
   ls data/annotations/reference/*.yaml | wc -l  # Should have 10+ files
   ```

2. **Use correct model**: Ensure you're using a trained model
   ```bash
   edm analyze track.mp3 --model outputs/training/best.pt
   ```

3. **Adjust novelty parameters**: Tune sensitivity (not yet exposed via CLI - TODO)

### Parallel processing errors

**Symptoms**: Crashes when using `--workers` > 1

**Solutions**:

1. **Reduce workers**:
   ```bash
   edm analyze tracks/ --workers 4  # Start low
   ```

2. **Check memory**: Ensure sufficient RAM
   ```bash
   free -h  # Linux/Mac
   ```

3. **Disable parallel processing**:
   ```bash
   edm analyze tracks/ --workers 1
   ```

---

## Training

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory, killed process

**Solutions** (try in order):

1. **Reduce batch size**:
   ```bash
   edm train data/annotations --batch-size 2
   ```

2. **Reduce segment duration**:
   ```bash
   edm train data/annotations --duration 15.0
   ```

3. **Use smaller backbone**:
   ```bash
   edm train data/annotations --backbone cnn
   ```

4. **Reduce workers**:
   ```bash
   edm train data/annotations --workers 2
   ```

5. **Use CPU only** (slow but works):
   ```bash
   edm train data/annotations --device cpu
   ```

### Training loss is NaN

**Cause**: Learning rate too high, causing gradient explosion

**Solution**: Reduce learning rate:

```bash
edm train data/annotations --lr 1e-5
```

Also check:
- Data quality (corrupted audio files)
- Annotation validity (check YAML syntax)

### Slow training

**Solutions**:

1. **Increase workers**:
   ```bash
   edm train data/annotations --workers 8
   ```

2. **Increase batch size** (if memory allows):
   ```bash
   edm train data/annotations --batch-size 8
   ```

3. **Use GPU**:
   ```bash
   edm train data/annotations --device cuda
   ```

4. **Reduce validation frequency**:
   ```bash
   edm train data/annotations --val-every 5
   ```

### Training not starting

**Checklist**:

1. **Verify annotations exist**:
   ```bash
   ls data/annotations/*.yaml | wc -l  # Should show count > 0
   ```

2. **Verify audio files exist**:
   ```bash
   python -c "
   import yaml
   from pathlib import Path
   ann = yaml.safe_load(Path('data/annotations/track1.yaml').read_text())
   print(f\"Audio: {ann['audio']['file']}\")
   print(f\"Exists: {Path(ann['audio']['file']).exists()}\")
   "
   ```

3. **Test dataset loading**:
   ```bash
   uv run python -c "
   from edm.training.dataset import EDMDataset
   from pathlib import Path
   dataset = EDMDataset(Path('data/annotations'))
   print(f'Loaded {len(dataset)} samples')
   print(dataset[0].keys())
   "
   ```

4. **Check disk space**:
   ```bash
   df -h outputs/  # Ensure sufficient space for checkpoints
   ```

### Model not improving

**Solutions**:

1. **Adjust learning rate**:
   ```bash
   # Lower for fine-tuning
   --lr 0.00005

   # Higher for training from scratch
   --lr 0.0003
   ```

2. **Increase epochs**:
   ```bash
   --epochs 100
   ```

3. **Adjust loss weights**:
   ```bash
   # Prioritize boundary detection
   --boundary-weight 2.0 \
   --energy-weight 0.3 \
   --beat-weight 0.3 \
   --label-weight 0.1
   ```

4. **Check for overfitting**:
   - If training loss decreases but validation loss increases
   - Solution: Add more training data or reduce model complexity

---

## Annotator Application

### Backend won't start

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version  # Needs 3.12+
   ```

2. **Install packages**:
   ```bash
   cd /path/to/edm
   uv sync
   ```

3. **Verify CLI works**:
   ```bash
   uv run edm-annotator --help
   ```

4. **Check audio directory**:
   ```bash
   ls $EDM_AUDIO_DIR  # Should list files
   ```

5. **Check port availability**:
   ```bash
   lsof -i :5000  # Should be empty
   # If occupied, kill process or use different port:
   uv run edm-annotator --port 5001
   ```

### Frontend won't start

**Solutions**:

1. **Check Node version**:
   ```bash
   node --version  # Needs 18+
   ```

2. **Install dependencies**:
   ```bash
   cd packages/edm-annotator/frontend
   pnpm install
   ```

3. **Clear cache**:
   ```bash
   cd packages/edm-annotator/frontend
   rm -rf node_modules .vite
   pnpm install
   ```

4. **Check port availability**:
   ```bash
   lsof -i :5174  # Should be empty
   ```

### No tracks showing in sidebar

**Solutions**:

1. **Verify audio directory**:
   ```bash
   echo $EDM_AUDIO_DIR
   ls $EDM_AUDIO_DIR  # Should list audio files
   ```

2. **Check supported formats**: .mp3, .flac, .wav, .m4a

3. **Check backend logs**: Look for file scanning errors in terminal

4. **Verify backend is running**:
   ```bash
   curl http://localhost:5000/health
   # Should return: {"status": "ok"}
   ```

### Waveform not loading

**Solutions**:

1. **Check browser console** (F12) for errors

2. **Verify backend is running**:
   ```bash
   curl http://localhost:5000/health
   ```

3. **Check CORS**: Backend should enable CORS in development mode

4. **Try different browser**: Some browsers have stricter security policies

5. **Check audio file size**: Very large files (>100MB) may be slow

### Audio won't play

**Solutions**:

1. **Check audio file validity**:
   ```bash
   ffplay audio/track.mp3  # Should play in terminal
   ```

2. **Check browser support**: Some browsers don't support certain codecs

3. **Check browser console** (F12) for errors

4. **Disable autoplay blocking**: Some browsers block autoplay - click Play button manually

5. **Try different format**: Convert to widely-supported format:
   ```bash
   ffmpeg -i track.flac track.mp3
   ```

### Boundaries not snapping to beats

**Cause**: Quantization disabled or incorrect BPM/downbeat

**Solutions**:

1. **Enable quantization**: Press **Q** key or toggle button

2. **Re-tap tempo**: Delete BPM and tap again while listening

3. **Adjust downbeat**: Press **D** at actual first beat of bar 1

### Save button disabled

**Cause**: Missing required data

**Checklist**:
- [ ] BPM is set (tap or manual)
- [ ] Downbeat is set (press D)
- [ ] At least one boundary exists
- [ ] All regions have labels

**Solution**: Complete missing items and save button will enable

---

## Experiment Tracking

### DVC: "File already tracked"

**Error**: `outputs/training/my_run is already tracked`

**Solution**: Remove old .dvc file first:

```bash
rm outputs/training/my_run.dvc
dvc add outputs/training/my_run
git add outputs/training/my_run.dvc .gitignore
git commit -m "update tracked model"
```

### DVC: "Cannot find checkpoint"

**Error**: Checkpoint file not found

**Solution**: Pull from DVC remote:

```bash
dvc pull outputs/training/my_run.dvc
```

If remote not configured:

```bash
dvc remote add -d myremote /path/to/remote/storage
dvc push
```

### DVC: "Cache corrupted"

**Solution**: Clear cache and re-pull:

```bash
dvc cache clean --force
dvc pull
```

### MLflow: "No models found"

**Solution**: Check experiment exists:

```bash
mlflow experiments list

# If missing, it will be created automatically on first training run
```

### MLflow: "Run not found"

**Solution**: Verify tracking URI matches:

```bash
python -c "import mlflow; print(mlflow.get_tracking_uri())"
# Should be: file:///path/to/edm/mlruns
```

Check that `MLFLOW_TRACKING_URI` environment variable is not set incorrectly.

### MLflow UI not showing runs

**Solution**: Start UI from correct directory:

```bash
cd /path/to/edm
mlflow ui --port 5001
# Open http://localhost:5001
```

---

## Performance Issues

### Analysis is very slow

**Solutions**:

1. **Use parallel processing**:
   ```bash
   edm analyze tracks/ --workers 8
   ```

2. **Check CPU usage**:
   ```bash
   top  # Look for high CPU processes
   ```

3. **Use GPU for structure analysis** (if model supports):
   ```bash
   edm analyze track.mp3 --model model.pt --device cuda
   ```

4. **Analyze smaller segments**: Use `--duration` for testing
   ```bash
   edm analyze track.mp3 --duration 60  # First 60 seconds only
   ```

### High memory usage

**Solutions**:

1. **Reduce workers**:
   ```bash
   edm analyze tracks/ --workers 2
   ```

2. **Process files individually**: Don't batch process large directories

3. **Close other applications**: Free up system memory

4. **Check for memory leaks**: Restart long-running processes

### Training using too much disk space

**Cause**: Checkpoints saved every epoch

**Solution**: Adjust checkpoint frequency (TODO: add flag)

Current behavior:
- Saves best model only by default
- Use `--save-every N` to save every N epochs (not yet implemented)

**Workaround**: Clean old checkpoints manually:

```bash
# Keep only best and last
cd outputs/training/run_name
ls -t *.pt | tail -n +3 | xargs rm
```

---

## Claude Code Integration

### Plugin not loading

**Solutions**:

1. **Check settings syntax**:
   ```bash
   cat ~/.claude/settings.json | jq .  # Should parse valid JSON
   ```

2. **Verify plugin is enabled**: Check `~/.claude/settings.json`

3. **Restart Claude Code session**: Exit and restart

4. **Check plugin logs**: Look in `~/.claude/logs/` if available

### Plugin conflicts

**Solutions**:

1. **Disable all plugins**: Remove from settings temporarily

2. **Enable one at a time**: Identify conflicting plugin

3. **Check documentation**: Look for known conflicts

4. **Update plugins**: Ensure using latest versions

### Performance issues with Claude Code

**Solutions**:

1. **Run audit**:
   ```bash
   # In Claude Code session
   /config
   ```

2. **Disable unused plugins**: Remove from settings

3. **Check for resource-intensive operations**: Look at hook logs

4. **Use lighter contexts**: Reduce injected context size

---

## Getting Help

### Before asking for help

1. **Check documentation**:
   - `docs/` directory for comprehensive guides
   - `README.md` files in each package
   - `--help` flag for CLI commands

2. **Search codebase**:
   ```bash
   grep -r "pattern" packages/
   rg "pattern" packages/  # Faster with ripgrep
   ```

3. **Check agent guide**: See `docs/agent-guide.md` for code locations

4. **Read test files**: Tests show usage examples
   ```bash
   # Find tests for a module
   find . -name "*test*.py" -path "*/tests/*"
   ```

5. **Check justfile**: See available commands
   ```bash
   just --list
   ```

### Debug logging

Enable verbose logging:

```bash
# CLI
edm analyze track.mp3 --log-level DEBUG

# Training
edm train data/annotations --log-level DEBUG

# Annotator
uv run edm-annotator --log-level DEBUG
```

Check logs:

```bash
# If logging to file
tail -f outputs/edm.log

# Or check console output
```

### Reporting issues

Include in bug reports:

1. **Environment**:
   ```bash
   python --version
   uv --version
   pip list | grep -E "(torch|librosa|edm)"
   ```

2. **Command used**: Exact command that caused error

3. **Error message**: Full error output (use `--log-level DEBUG`)

4. **Steps to reproduce**: Minimal example that triggers issue

5. **Expected vs actual behavior**: What you expected to happen

6. **Context**: Operating system, hardware (GPU/CPU), data characteristics

### Resources

- **Project documentation**: `docs/`
- **Architecture overview**: `docs/architecture.md`
- **Development guide**: `docs/development.md`
- **Code style guides**: `docs/development/code-style-python.md`, `docs/development/code-style-javascript.md`
- **Agent navigation**: `docs/agent-guide.md`

---

## Common Workflow Issues

### Annotation workflow interrupted

**Problem**: Lost work after browser crash

**Solution**: Annotations auto-save to localStorage, reload page to recover

**Prevention**: Save frequently (Ctrl+S or Save button)

### Can't find saved annotations

**Location**: Check annotation directory:

```bash
echo $EDM_ANNOTATION_DIR  # Default: data/annotations
ls $EDM_ANNOTATION_DIR/reference/
```

### Training with old annotations

**Problem**: Model trained on outdated annotations

**Solution**: Check annotation timestamps:

```bash
ls -lt data/annotations/reference/ | head
```

Ensure recent annotations are included in training directory.

### Model inference doesn't match training

**Problem**: Different results between training and inference

**Checklist**:
- Using same model file (check path)
- Using same audio preprocessing (sample rate, hop length)
- Using same backbone architecture
- Model was loaded correctly (check for warnings)

### Git merge conflicts in annotations

**Problem**: Multiple people annotating same track

**Solution**: Annotations are YAML files, resolve manually:

```bash
# View conflict
cat data/annotations/reference/track.yaml

# Resolve using one version
git checkout --theirs data/annotations/reference/track.yaml
# Or
git checkout --ours data/annotations/reference/track.yaml

# Or merge manually in editor
```

**Prevention**: Coordinate annotation assignments, use different tracks
