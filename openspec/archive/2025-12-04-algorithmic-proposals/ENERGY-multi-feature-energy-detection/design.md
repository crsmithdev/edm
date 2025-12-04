# [ENERGY] Multi-Feature Energy Detection - Design

## Approach

Extend energy-based structure detection with multi-feature acoustic analysis using librosa. Implement in two stages:
1. Enhanced `EnergyDetector` with multi-feature scoring
2. New `HybridDetector` combining MSAF boundaries with multi-feature labeling

## Feature Extraction

### Core Features (All using hop_length=512, frame_length=2048)

```python
# Already implemented
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

# New features
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

# Band-specific energy
stft = librosa.stft(y, hop_length=512, n_fft=2048)
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
bass_mask = (freqs >= 20) & (freqs < 250)
mid_mask = (freqs >= 250) & (freqs < 4000)
high_mask = (freqs >= 4000) & (freqs <= sr/2)

bass_energy = np.sum(np.abs(stft[bass_mask, :]), axis=0)
mid_energy = np.sum(np.abs(stft[mid_mask, :]), axis=0)
high_energy = np.sum(np.abs(stft[high_mask, :]), axis=0)
```

### Normalization
Each feature normalized to [0, 1] independently:
```python
feature_norm = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-8)
```

### Smoothing
Apply median filter (size=21) to reduce frame-level noise, consistent with current RMS smoothing.

## Multi-Feature Section Scoring

### Drop Detection
```python
# Weighted feature score
drop_score = (
    0.30 * rms_norm +           # Overall loudness
    0.25 * bass_ratio +         # Bass presence
    0.20 * spectral_contrast +  # Timbral variation
    0.15 * spectral_centroid +  # Brightness
    0.10 * onset_strength       # Transient density
)

# Threshold: drop_score > 0.65
# Confidence: min(0.8 + (drop_score - 0.65) * 0.5, 0.99)
```

### Buildup Detection
```python
# Rising energy + increasing features
rms_gradient = np.gradient(rms_smooth)
onset_gradient = np.gradient(onset_smooth)

buildup_score = (
    0.40 * rms_gradient_norm +
    0.30 * onset_gradient_norm +
    0.20 * spectral_centroid_norm +
    0.10 * mid_energy_ratio
)

# Threshold: buildup_score > 0.55 AND positive gradients
```

### Breakdown Detection
```python
breakdown_score = (
    0.40 * (1 - rms_norm) +        # Low energy
    0.30 * (1 - onset_strength) +  # Low transients
    0.20 * (1 - bass_ratio) +      # Less bass
    0.10 * spectral_stability      # Stable timbre
)

# Threshold: breakdown_score > 0.60
```

### Intro/Outro
Keep existing position-based heuristics (first/last sections).

## Implementation Details

### EnergyDetector Changes

**File**: `src/edm/analysis/structure_detector.py`

**Method**: `detect()`
```python
def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
    # Load audio
    y, sr = librosa.load(str(filepath), sr=sr, mono=True)
    duration = len(y) / sr

    # Extract features (new method)
    features = self._extract_features(y, sr)

    # Detect boundaries using multi-feature analysis
    boundaries = self._detect_boundaries_multifeature(features, sr, duration)

    # Label sections with multi-feature scoring
    sections = self._boundaries_to_sections_multifeature(
        boundaries, features, sr, duration
    )

    # Merge short sections
    sections = self._merge_short_sections(sections)

    return sections
```

**New method**: `_extract_features()`
```python
def _extract_features(self, y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Extract all acoustic features for structure analysis."""
    hop_length = 512
    frame_length = 2048

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=0)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Band-specific energy
    stft = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    bass_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = (freqs >= 4000)

    total_energy = np.sum(np.abs(stft), axis=0) + 1e-8
    bass_ratio = np.sum(np.abs(stft[bass_mask, :]), axis=0) / total_energy
    mid_ratio = np.sum(np.abs(stft[mid_mask, :]), axis=0) / total_energy
    high_ratio = np.sum(np.abs(stft[high_mask, :]), axis=0) / total_energy

    # Smooth and normalize all features
    from scipy.ndimage import median_filter

    features = {
        'rms': self._normalize(median_filter(rms, size=21)),
        'spectral_centroid': self._normalize(median_filter(spectral_centroid, size=21)),
        'spectral_contrast': self._normalize(median_filter(spectral_contrast_mean, size=21)),
        'onset_strength': self._normalize(median_filter(onset_env, size=21)),
        'bass_ratio': self._normalize(median_filter(bass_ratio, size=21)),
        'mid_ratio': self._normalize(median_filter(mid_ratio, size=21)),
        'high_ratio': self._normalize(median_filter(high_ratio, size=21)),
    }

    return features

def _normalize(self, arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
```

**Updated method**: `_boundaries_to_sections_multifeature()`
Replace single RMS energy with feature-based scoring for each section.

### HybridDetector (New Class)

**File**: `src/edm/analysis/structure_detector.py`

```python
class HybridDetector:
    """MSAF boundaries + multi-feature energy labeling."""

    def __init__(self):
        self._msaf = MSAFDetector()
        self._energy = EnergyDetector()  # Uses multi-feature version

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        # Get MSAF boundaries
        msaf_sections = self._msaf.detect(filepath, sr)
        boundaries = [s.start_time for s in msaf_sections] + [msaf_sections[-1].end_time]

        # Extract features
        y, sr_actual = librosa.load(str(filepath), sr=sr, mono=True)
        features = self._energy._extract_features(y, sr_actual)

        # Label segments using multi-feature analysis
        sections = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Get feature values for this segment
            start_frame = int(start * sr_actual / 512)
            end_frame = int(end * sr_actual / 512)

            segment_features = {
                k: np.mean(v[start_frame:end_frame])
                for k, v in features.items()
            }

            # Score and label
            label, confidence = self._score_segment(segment_features, i, len(boundaries) - 1)

            sections.append(DetectedSection(
                start_time=start,
                end_time=end,
                label=label,
                confidence=confidence,
            ))

        return sections

    def _score_segment(self, features: dict, idx: int, total: int) -> tuple[str, float]:
        """Score segment and assign label based on multi-feature analysis."""
        # Intro/outro heuristics
        if idx == 0:
            return "intro", 0.9
        if idx == total - 1:
            return "outro", 0.9

        # Drop scoring
        drop_score = (
            0.30 * features['rms'] +
            0.25 * features['bass_ratio'] +
            0.20 * features['spectral_contrast'] +
            0.15 * features['spectral_centroid'] +
            0.10 * features['onset_strength']
        )

        if drop_score > 0.65:
            confidence = min(0.8 + (drop_score - 0.65) * 0.5, 0.99)
            return "drop", float(confidence)

        # Breakdown scoring
        breakdown_score = (
            0.40 * (1 - features['rms']) +
            0.30 * (1 - features['onset_strength']) +
            0.20 * (1 - features['bass_ratio']) +
            0.10 * (1 - features['spectral_contrast'])
        )

        if breakdown_score > 0.60:
            return "breakdown", 0.75

        # Default: other
        return "other", 0.5
```

### Detector Selection

**File**: `src/edm/analysis/structure.py`

Update `analyze_structure()` to accept `detector="hybrid"`:

```python
def analyze_structure(
    filepath: Path,
    detector: str = "auto",  # auto|msaf|energy|hybrid
    bpm: float | None = None,
    include_bars: bool = True,
    time_signature: TimeSignature = (4, 4),
) -> StructureResult:
    ...
```

Update `get_detector()`:

```python
def get_detector(detector_type: str) -> StructureDetector:
    if detector_type == "energy":
        return EnergyDetector()
    if detector_type == "hybrid":
        return HybridDetector()
    if detector_type in ("msaf", "auto"):
        return MSAFDetector()
    raise ValueError(f"Unknown detector type: {detector_type}")
```

## Testing Strategy

### Unit Tests
- Feature extraction produces expected shapes
- Normalization handles edge cases (zero variance)
- Scoring functions return valid ranges [0, 1]

### Integration Tests
- Multi-feature detector runs without errors
- Hybrid detector produces valid sections (no overlaps, full coverage)
- Performance remains <30s/track

### Evaluation
- Run `/evaluate` on annotated tracks with each detector:
  - `detector=energy` (multi-feature)
  - `detector=hybrid`
  - Compare to baseline (current RMS-only)
- Measure boundary F1, drop precision/recall

## Performance Considerations

### Compute Cost
- STFT already computed for RMS → reuse for band energy (minimal overhead)
- Spectral features use same hop_length → aligned frames
- Feature extraction adds ~5-10s per track (estimate)

### Optimization Opportunities
- Cache STFT computation across features
- Vectorize feature scoring (avoid loops)
- Use lower sample rate (22050 Hz) for feature extraction

## Risks

### Parameter Sensitivity
Feature weights and thresholds chosen heuristically. May need tuning per dataset.

**Mitigation**: Track evaluation metrics, iterate on weights.

### Library Dependency
Relies heavily on librosa. Alternative: Essentia for faster extraction.

**Mitigation**: Start with librosa, profile performance, switch if needed.

### Complexity
More features = more code paths to maintain.

**Mitigation**: Keep feature extraction modular, well-tested.
