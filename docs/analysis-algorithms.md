# Analysis Algorithms

Technical details of BPM and structure detection algorithms.

## BPM Detection

### beat_this (Neural Network)

Primary BPM detector using the beat_this library (ISMIR 2024).

**Algorithm**:
1. Load audio at 22050 Hz sample rate
2. Compute mel spectrogram
3. Run neural network beat tracker
4. Extract beat timestamps
5. Calculate inter-beat intervals (IBIs)
6. Estimate tempo from median IBI
7. Apply tempo multiplicity correction

**Tempo Multiplicity**:
- Raw tempo may be half or double actual BPM
- Adjust to preferred EDM range (120-150 BPM)
- If tempo < 100: double it
- If tempo > 180: halve it

**Confidence Score**:
```python
# Based on IBI consistency
ibi_std = np.std(inter_beat_intervals)
confidence = 1.0 - min(ibi_std / median_ibi, 1.0)
```

**Performance**:
- Time: 5-10s per track
- Memory: ~500MB (GPU) or ~200MB (CPU)
- Accuracy: ~95% within ±2 BPM for EDM

### librosa (Spectral Flux)

Fallback BPM detector using librosa's tempo estimation.

**Algorithm**:
1. Load audio at 22050 Hz
2. Compute onset strength envelope
3. Calculate autocorrelation of onset envelope
4. Find peaks in autocorrelation → tempo candidates
5. Apply tempogram-based refinement
6. Return highest-confidence estimate

**Tempo Range**:
- Default: 60-180 BPM
- Adjusted to prefer EDM range when multiple candidates

**Performance**:
- Time: 2-3s per track
- Memory: ~100MB
- Accuracy: ~85% within ±2 BPM for EDM

### Metadata Extraction

Fastest method - read BPM from file tags.

**Supported Formats**:
- ID3v2 (MP3): `TBPM` frame
- Vorbis (FLAC, OGG): `BPM` or `TEMPO` tag
- MP4/M4A: `tmpo` atom

**Validation**:
- BPM must be numeric
- Range: 60-200 (reject outliers)
- Missing tags return None

## Structure Detection

### MSAF (Music Structure Analysis Framework)

Primary structure detector using MSAF library.

**Boundary Detection**:

1. **Feature Extraction**:
   - Compute CQT (Constant-Q Transform)
   - Extract MFCCs (Mel-Frequency Cepstral Coefficients)
   - Build self-similarity matrix

2. **Novelty Curve**:
   - Compute spectral flux along diagonal
   - Apply Gaussian smoothing
   - Peak picking for boundary candidates

3. **Boundary Refinement**:
   - Minimum section duration: 8 seconds
   - Merge adjacent similar sections
   - Ensure full track coverage

**Label Assignment**:

Energy-based mapping to EDM labels:

```python
def assign_label(section, energy_percentile):
    if energy_percentile > 0.75:
        return "drop"
    elif energy_percentile > 0.5:
        return "buildup" if is_rising else "breakdown"
    elif is_first_section:
        return "intro"
    elif is_last_section:
        return "outro"
    else:
        return "breakdown"
```

**Energy Calculation**:
- RMS energy per frame
- Spectral contrast (high vs low frequency energy)
- Weighted combination for EDM characteristics

**Performance**:
- Time: 3-5s per track
- Memory: ~200MB
- Boundary accuracy: ~70% within ±2s

### Energy Detector (Rule-Based)

Fallback structure detector using RMS energy analysis.

**Algorithm**:

1. **Energy Extraction**:
   ```python
   rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
   ```

2. **Boundary Detection**:
   - Compute energy gradient: `gradient = np.diff(smoothed_rms)`
   - Find peaks in absolute gradient
   - Apply minimum section duration filter (8s)

3. **Label Assignment**:
   - Compute energy percentiles per section
   - Threshold-based labeling:
     - Top 25%: drop
     - Middle 50%: breakdown/buildup (based on direction)
     - First section: intro
     - Last section: outro

4. **Post-Processing**:
   - Ensure no gaps between sections
   - Merge very short sections (<4s)
   - Validate full track coverage

**Performance**:
- Time: 1-2s per track
- Memory: ~100MB
- Boundary accuracy: ~60% within ±2s

## Bar Calculation

Convert time positions to musical bars.

**Formula**:
```python
def time_to_bars(time_seconds: float, bpm: float, time_signature: tuple[int, int]) -> float:
    beats_per_bar = time_signature[0]
    beats = time_seconds * bpm / 60.0
    bars = beats / beats_per_bar
    return bars + 1.0  # 1-indexed
```

**Conventions**:
- Bar 1 starts at time 0
- Fractional bars supported (16.5 = halfway through bar 17)
- Default time signature: 4/4

**Inverse**:
```python
def bars_to_time(bar: float, bpm: float, time_signature: tuple[int, int]) -> float:
    beats_per_bar = time_signature[0]
    beats = (bar - 1.0) * beats_per_bar
    return beats * 60.0 / bpm
```

## Algorithm Selection

### When to Use Each

| Scenario | BPM | Structure |
|----------|-----|-----------|
| EDM tracks, accuracy critical | beat_this | MSAF |
| Speed priority | metadata/librosa | energy |
| Non-EDM music | librosa | MSAF |
| Batch processing | metadata first | energy |

### Automatic Selection

**BPM** (`analyze_bpm`):
1. Try metadata (instant)
2. Fall back to beat_this (accurate)
3. Fall back to librosa (if beat_this unavailable)

**Structure** (`analyze_structure` with `detector="auto"`):
1. Try MSAF (if available)
2. Fall back to energy (always available)

## References

- beat_this: [ISMIR 2024 Paper](https://archives.ismir.net/ismir2024/)
- MSAF: [Nieto & Bello, ISMIR 2016](https://doi.org/10.5281/zenodo.259148)
- librosa: [McFee et al., SciPy 2015](https://doi.org/10.25080/Majora-7b98e3ed-003)
