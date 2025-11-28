# Change: Add Structure Analysis

## Why

The current `analyze_structure()` function is a placeholder returning hardcoded sections regardless of audio content. Structure analysis is a core EDM analysis capability needed for:

- **DJ mixing**: Identify intro/outro for clean transitions, locate drops for energy management
- **Track organization**: Categorize tracks by structure complexity
- **Cue point suggestions**: Automatically mark drop locations, breakdown starts
- **Mix planning**: Understand track flow for set construction

Without real structure detection, the library cannot fulfill its primary purpose of comprehensive EDM track analysis.

## What Changes

Replace the placeholder implementation with MSAF-based structure detection:

1. **Integrate MSAF** - Music Structure Analysis Framework (Nieto & Bello, ISMIR 2016)
   - Multiple boundary detection algorithms (Laplacian, Foote, etc.)
   - Multiple labeling algorithms for segment classification
   - Lightweight dependencies (librosa, scipy, scikit-learn - no PyTorch/madmom)
   - CPU-only, no GPU requirements

2. **Add energy-based fallback** - Rule-based drop detection using librosa
   - RMS energy analysis for high-energy section detection
   - Spectral contrast for bass-heavy drop identification
   - Onset strength patterns for buildup detection
   - Used when MSAF unavailable or for validation

3. **Implement section mapping** - Convert MSAF output to EDM-specific labels
   - High-energy segments → drop
   - Low-energy segments after drops → breakdown
   - Rising-energy segments before drops → buildup
   - First/last segments → intro/outro

4. **Add evaluation framework** - Structure accuracy metrics
   - Boundary tolerance evaluation (±2 seconds)
   - Section classification accuracy
   - Integration with existing `edm evaluate` command

5. **Update CLI** - Structure-specific options
   - `--types structure` for structure-only analysis
   - `--structure-detector` to select detection method (msaf/energy/auto)
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
  - Add: `msaf` (music structure analysis framework)
  - Existing: `librosa` (energy-based fallback, also used by MSAF)
- **User experience**: `edm analyze track.mp3 --types structure` returns real section data
- **Performance**: < 30 seconds per track (accuracy-first tradeoff)

## Design Decisions

### Why MSAF?

1. **Established framework** - Academic-grade (ISMIR 2016), well-documented
2. **Lightweight dependencies** - librosa, scipy, scikit-learn (no PyTorch, no madmom)
3. **Multiple algorithms** - Can experiment with different boundary/labeling methods
4. **MIT license** - Compatible with project licensing
5. **No GPU required** - Works on any system without CUDA complexity

### Why not Allin1?

- Requires PyTorch with specific version constraints
- Depends on madmom which is unmaintained and has Python 3.10+ issues
- NATTEN dependency has complex GPU/PyTorch version matrix
- Sequential processing only, cannot batch
- Heavy installation (~2GB+ with PyTorch)

### Why not custom training?

- Requires 1000+ labeled EDM tracks with section boundaries
- Significant development cycle for competitive accuracy
- MSAF provides immediate value with proven algorithms

### Why energy-based fallback?

- MSAF may fail on unusual tracks
- Drop detection via energy analysis is well-established for EDM
- Provides validation/ensemble opportunity
- Zero additional dependencies (uses librosa)

### Section Label Mapping Strategy

MSAF returns segment boundaries with cluster labels. We map to EDM labels using energy analysis:

| Segment Characteristic | EDM Label | Detection Method |
|------------------------|-----------|------------------|
| First segment | intro | Position-based |
| High sustained energy | drop | RMS energy threshold |
| Low energy after drop | breakdown | Energy dip detection |
| Rising energy before drop | buildup | Energy gradient |
| Last segment | outro | Position-based |

### Accuracy Targets

Per project.md constraints:
- Drop detection: Precision >90%, Recall >85%
- Boundary tolerance: ±2 seconds
- Processing time: < 30 seconds per track
