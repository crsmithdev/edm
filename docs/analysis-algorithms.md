# Analysis Algorithms

Technical details of BPM detection, structure analysis, and bar calculation algorithms.

## BPM Detection

### Fallback Chain

BPM detection uses a cascading strategy that tries multiple sources in order of speed:

1. **Metadata** (instant) - ID3/FLAC/MP4 tags
2. **beat_this** (5-10s) - Neural network beat tracker
3. **librosa** (2-3s) - Spectral flux tempo estimation

Control via CLI:
- `--ignore-metadata`: Skip metadata, force computation
- Strategy automatically falls back if earlier methods fail

Implementation: `src/edm/analysis/bpm.py:36`

### beat_this Neural Tracker

**Algorithm**: Neural network-based beat and downbeat tracking (ISMIR 2024)

**Process**:
1. Load audio and resample to model's expected sample rate
2. Run inference to detect beat positions (timestamps)
3. Calculate BPM from median beat interval: `60.0 / median(intervals)`
4. Calculate confidence from interval consistency: `1.0 - (std / median)`

**Tempo Multiplicity**:
- Detects half-time and double-time alternatives (0.5x, 2.0x BPM)
- Adjusts to EDM range (120-150 BPM preferred)
- Returns alternatives in `BPMResult.alternatives`

**Trade-offs**:
- Pros: Highest accuracy for EDM, handles complex rhythms
- Cons: Slow initialization (~5s first run), GPU memory usage

Implementation: `src/edm/analysis/bpm_detector.py:33`

### librosa Fallback

**Algorithm**: Spectral flux + autocorrelation tempo estimation

**Process**:
1. Extract onset envelope from spectral flux
2. Compute autocorrelation of onset strength
3. Find dominant tempo from autocorrelation peaks
4. Return static tempo estimate

**Trade-offs**:
- Pros: Fast, no GPU required, well-tested
- Cons: Less accurate for EDM (especially variable tempo)

Implementation: `src/edm/analysis/bpm_detector.py:104`

## Structure Detection

### Detector Selection

Structure analysis supports two detector types:

- **msaf** (default): MSAF boundary detection + energy-based labeling
- **energy**: Rule-based energy analysis (no ML)

CLI control: `--structure-detector msaf|energy|auto`

Implementation: `src/edm/analysis/structure.py:94`

### MSAF Detector

**Boundary Detection**:
- Spectral Flux algorithm (Nieto & Bello, ISMIR 2016)
- Detects changes in spectral energy distribution
- Returns boundary timestamps (e.g., `[0.0, 30.5, 90.2, 180.0]`)

**Labeling**:
- MSAF provides generic labels (A, B, C, etc.)
- Energy thresholds map to EDM terminology

**Energy-Based EDM Label Mapping**:
```python
# RMS energy percentiles
high_energy = 75th percentile
low_energy = 25th percentile

if energy > high_energy:
    label = "drop"
elif energy < low_energy:
    label = "breakdown"
elif is_rising(energy):
    label = "buildup"
elif position < 0.2 * duration:
    label = "intro"
else:
    label = "outro"
```

**Confidence Calculation**:
- Based on energy characteristics strength
- High energy consistency → high confidence
- Ambiguous sections → lower confidence

Implementation: `src/edm/analysis/structure_detector.py:63`

### Energy Detector

**Algorithm**: Rule-based analysis using RMS energy and spectral contrast

**Boundary Detection**:
1. Calculate RMS energy over sliding window (2048 samples)
2. Compute energy gradient: `gradient = diff(energy)`
3. Find peaks in gradient (threshold: 75th percentile)
4. Filter boundaries by minimum section duration (8 seconds)

**Section Labeling**:
```python
# Energy thresholds (percentiles of track energy)
high = 80th percentile
medium = 50th percentile
low = 20th percentile

# Position thresholds
intro_end = 0.15 * duration
outro_start = 0.85 * duration

if position < intro_end:
    return "intro"
elif position > outro_start:
    return "outro"
elif energy > high and spectral_contrast > medium:
    return "drop"
elif energy < low:
    return "breakdown"
elif gradient > 0:  # rising energy
    return "buildup"
else:
    return "breakdown"
```

**Full Coverage Guarantee**:
- Algorithm ensures no gaps between sections
- Last section extends to track end
- Merges adjacent sections with same label

**Trade-offs**:
- Pros: Fast, no dependencies, consistent results
- Cons: Less accurate than MSAF, relies on energy patterns

Implementation: `src/edm/analysis/structure_detector.py:260`

### Energy Thresholds

Both detectors use percentile-based energy thresholds:

```python
energy_rms = librosa.feature.rms(y=audio)[0]
percentiles = np.percentile(energy_rms, [25, 50, 75, 80])
low, medium, high, very_high = percentiles
```

**Rationale**:
- Percentiles adapt to track's energy distribution
- Avoids hard-coded thresholds that fail on quiet/loud tracks
- EDM tracks often have extreme dynamic range

## Bar Calculation

### Time to Bars Conversion

**Formula**:
```python
bar_position = (time_seconds * bpm / 60.0) / time_signature.beats_per_bar
```

**Example** (4/4 time, 128 BPM):
```python
time = 30.0 seconds
bar = (30.0 * 128 / 60.0) / 4
    = 64.0 beats / 4
    = 16.0 bars (1-indexed: bar 16)
```

**Time Signature Support**:
- Default: 4/4 (4 beats per bar)
- Supported: 3/4, 6/8, 5/4, etc.
- Parsed from metadata or user-specified

Implementation: `src/edm/analysis/bars.py:15`

### Bar Numbering

Bars are **1-indexed** to match DJ software conventions:
- Bar 1 = first bar of track (time 0.0s)
- Bar 17 = 17th bar (64 beats into track at 128 BPM)

This differs from 0-indexed programming but matches musician expectations.

### Fractional Bars

Bar positions can be fractional for precision:
- `16.5` = halfway through bar 17
- `16.75` = 3/4 through bar 17

Useful for:
- Section boundaries that don't align to bar starts
- Drop analysis (e.g., "drop hits on the 1" = bar.0)

### Graceful Degradation

When BPM is unavailable (metadata missing, detection failed):
- `start_bar`, `end_bar`, `bar_count` = `None`
- Time-based fields (`start_time`, `end_time`) still populated
- Allows partial results instead of complete failure

Implementation: `src/edm/analysis/structure.py:127`

## Beat Grid (Future)

Placeholder support exists for beat grid anchoring:

**Concept**:
- Detect first downbeat timestamp
- Align all bars to downbeat (not time 0.0)
- Ensures bar.0 = musical downbeat

**Current Status**:
- `beat_grid` parameter accepted but unused
- Bars calculated from BPM + time signature only
- Downbeat assumed at time 0.0

**Future Implementation**:
- Use beat_this downbeat detection
- Adjust bar calculations: `bar = (time - downbeat) * bpm / 60.0 / beats_per_bar`
- Improve accuracy for tracks with intro silence

Related files: `src/edm/analysis/beat_grid.py` (stub), `src/edm/analysis/bars.py:15`

## Performance Characteristics

| Operation | Method | Time | Memory | GPU |
|-----------|--------|------|--------|-----|
| BPM (metadata) | ID3/FLAC tags | <10ms | <1MB | No |
| BPM (beat_this) | Neural network | 5-10s | ~500MB | Optional |
| BPM (librosa) | Spectral flux | 2-3s | ~100MB | No |
| Structure (MSAF) | Spectral analysis | 3-5s | ~200MB | No |
| Structure (energy) | RMS energy | 1-2s | ~100MB | No |
| Bar calculation | Math | <1ms | <1KB | No |

**Constraints** (from architecture.md):
- Max processing time: 30s per track
- Memory limit: 4GB
- Consumer GPU support: CUDA optional

## Algorithm Selection Guide

### When to Use beat_this
- EDM tracks with complex rhythms
- Accuracy critical (DJ mixes, tempo mapping)
- GPU available (faster inference)

### When to Use librosa
- Speed priority over accuracy
- CPU-only environment
- Non-EDM tracks (rock, classical)

### When to Use MSAF
- Accurate section boundaries needed
- Track has clear structural changes
- Labeling less critical than boundaries

### When to Use Energy Detector
- Speed critical (real-time analysis)
- MSAF unavailable (dependency issues)
- Energy-based sections sufficient (drop/breakdown focus)
