# [BEATSYNC] Beat-Synchronized Energy Analysis

**Status**: draft
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

Current `EnergyDetector` calculates RMS energy using fixed frame sizes (512-sample hop, ~23ms at 22050Hz). This creates:
- **Temporal smearing**: Energy changes between beats get averaged
- **Phase misalignment**: Frames not aligned to beat grid
- **Noise sensitivity**: High-frequency noise inflates RMS between beats

Example problem:
```
Frame 1: [kick + quiet]  → RMS = 0.5
Frame 2: [quiet + kick]  → RMS = 0.5
```
Both frames have same RMS but different musical content.

Beat-synchronized would give:
```
Beat 1: [kick] → RMS = 0.8
Beat 2: [kick] → RMS = 0.8
```

Cleaner signal, better boundary detection.

## What

Calculate energy per beat/bar instead of fixed time frames.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - EnergyDetector._detect_boundaries()
- `src/edm/analysis/beat_detector.py` - Reuse existing beat grid
- `openspec/specs/analysis/spec.md` - Document beat-sync energy

**Approach:**
1. Detect beat grid first (using existing `detect_beats()`)
2. For each beat interval, calculate RMS energy
3. Smooth beat-level energy (median filter over 4-8 beats)
4. Detect boundaries at beat-level energy changes
5. Snap boundaries to nearest beat time

**Benefits over fixed frames:**
- Energy curve aligned to musical structure
- Reduced noise from inter-beat regions
- More stable boundary detection
- Bar-aligned boundaries by default

## Impact

**Breaking changes:** None (internal change to EnergyDetector)

**Benefits:**
- Cleaner energy signal (less noise)
- More musically meaningful boundaries
- Natural bar alignment
- Better drop detection (drops often start on downbeats)

**Risks:**
- Depends on accurate beat detection
- Fails if BPM detection fails (need fallback)
- May miss boundaries that occur mid-beat
- Slightly slower (requires beat detection first)

**Performance:**
- Beat detection: ~2-3s per track (already done for bar calculations)
- Beat-level RMS: faster than frame-level (fewer samples)
- Net: similar or faster than current approach

**Mitigation:**
- Fallback to frame-based if beat detection fails
- Add sub-beat resolution for transitions
- Make beat-sync optional parameter
- Validate on tracks with tempo changes

**Dependencies:**
- Requires `src/edm/analysis/beat_detector.py:detect_beats()`
- Reuses existing beat tracking infrastructure
