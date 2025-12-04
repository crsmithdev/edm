# [MULTIENG] Multi-Feature Energy Detection

**Status**: draft
**Created**: 2025-12-04
**Updated**: 2025-12-04

## Why

Current `EnergyDetector` uses only RMS energy for structure detection, which conflates volume with musical intensity. EDM drops often feature:
- High spectral brightness (spectral centroid)
- Dense high-frequency content (spectral rolloff)
- Increased percussive elements (zero-crossing rate)
- Rich harmonic content (chroma energy)

Single-feature RMS cannot distinguish between:
- Loud breakdowns (high RMS, low brightness)
- Quiet but intense drops (moderate RMS, high brightness/percussion)

## What

Extend `EnergyDetector` with multi-dimensional energy analysis combining RMS with spectral features.

**Affected files:**
- `src/edm/analysis/structure_detector.py` - EnergyDetector class
- `openspec/specs/analysis/spec.md` - Structure detection spec

**New features:**
- **Spectral centroid**: Brightness (Hz) - drops typically >3kHz
- **Spectral rolloff**: High-freq content (Hz) - correlates with cymbal/hi-hat density
- **Zero-crossing rate**: Noise/percussion indicator
- **Chroma energy**: Harmonic richness

**Approach:**
- Calculate all features per frame alongside RMS
- Normalize and combine into composite energy score
- Weight features: `energy = 0.4*RMS + 0.3*centroid + 0.2*rolloff + 0.1*zcr`
- Use composite for boundary detection and labeling

## Impact

**Breaking changes:** None (internal enhancement)

**Benefits:**
- More robust drop/breakdown distinction
- Better handling of quiet-but-intense sections
- Reduced false positives from volume-only detection

**Risks:**
- Increased computation (~2x slower)
- More parameters to tune (feature weights)
- Features may correlate differently across subgenres

**Performance:**
- Current: ~5s per track (RMS only)
- Estimated: ~10s per track (5 features)
- Within 30s target constraint

**Mitigation:**
- Make feature set configurable
- Provide genre-specific presets
- Add feature importance analysis for debugging
