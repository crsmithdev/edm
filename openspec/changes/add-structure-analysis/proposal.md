# Change: Add Structure Analysis

## Why

The current `analyze_structure()` function is a placeholder returning hardcoded sections regardless of audio content. Structure analysis is a core EDM analysis capability needed for:

- **DJ mixing**: Identify intro/outro for clean transitions, locate drops for energy management
- **Track organization**: Categorize tracks by structure complexity
- **Cue point suggestions**: Automatically mark drop locations, breakdown starts
- **Mix planning**: Understand track flow for set construction

Without real structure detection, the library cannot fulfill its primary purpose of comprehensive EDM track analysis.

## What Changes

Replace the placeholder implementation with ML-based structure detection:

1. **Integrate Allin1 model** - State-of-the-art music structure analysis model (2023)
   - Pre-trained weights for section boundary detection
   - Maps generic sections (chorus/verse/bridge) to EDM terminology (drop/breakdown/buildup)
   - GPU acceleration when available, CPU fallback

2. **Add energy-based fallback** - Rule-based drop detection using librosa
   - RMS energy analysis for high-energy section detection
   - Spectral contrast for bass-heavy drop identification
   - Onset strength patterns for buildup detection
   - Used when Allin1 unavailable or for validation

3. **Implement section mapping** - Convert model output to EDM-specific labels
   - chorus → drop (high energy, full arrangement)
   - bridge → breakdown (reduced energy, melodic focus)
   - verse → buildup (building energy, tension)
   - intro/outro → preserved as-is

4. **Add evaluation framework** - Structure accuracy metrics
   - Boundary tolerance evaluation (±2 seconds)
   - Section classification accuracy
   - Integration with existing `edm evaluate` command

5. **Update CLI** - Structure-specific options
   - `--types structure` for structure-only analysis
   - `--structure-model` to select detection method (allin1/energy/auto)
   - JSON output includes section boundaries with confidence

## Impact

- **Affected specs**: `analysis` (new structure detection requirements)
- **Affected code**:
  - Modified: `src/edm/analysis/structure.py` (replace placeholder)
  - New: `src/edm/analysis/structure_detector.py` (detection implementations)
  - Modified: `src/edm/analysis/__init__.py` (exports)
  - Modified: `src/edm/config.py` (structure detection settings)
  - Modified: `src/edm/cli/` (structure options)
  - New: `src/edm/evaluation/structure.py` (accuracy evaluation)
- **Dependencies**:
  - Add: `allin1` (music structure analysis model)
  - Existing: `librosa` (energy-based fallback)
- **User experience**: `edm analyze track.mp3 --types structure` returns real section data
- **Performance**: < 30 seconds per track (accuracy-first tradeoff)

## Design Decisions

### Why Allin1?

1. **State-of-the-art accuracy** - Recent (2023) transformer-based model outperforming older approaches
2. **Pre-trained weights** - No training infrastructure needed
3. **Maintained** - Active development, pip-installable
4. **Extensible** - Can fine-tune on EDM data later if generic model insufficient

### Why not custom training?

- Requires 1000+ labeled EDM tracks with section boundaries
- 3-6 month development cycle for competitive accuracy
- Significant GPU infrastructure for training
- Allin1 provides immediate value; fine-tuning can come later

### Why energy-based fallback?

- Allin1 may not be installed (large dependency)
- GPU memory constraints on some systems
- Drop detection via energy analysis is well-established for EDM
- Provides validation/ensemble opportunity

### Section Label Mapping

| Allin1 Label | EDM Label | Rationale |
|--------------|-----------|-----------|
| intro | intro | Direct mapping |
| verse | buildup | Pre-drop tension building |
| chorus | drop | High energy payoff section |
| bridge | breakdown | Reduced energy, melodic |
| outro | outro | Direct mapping |
| instrumental | varies | Context-dependent |

### Accuracy Targets

Per project.md constraints:
- Drop detection: Precision >90%, Recall >85%
- Boundary tolerance: ±2 seconds
- Processing time: < 30 seconds per track
