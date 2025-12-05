# [MLPIVOT] Technical Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Input                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Feature Extraction                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │  Mel Spectrogram│  │  MERT Embeddings│  │  Beat Features │     │
│  │  (input repr)   │  │  (pretrained)  │  │  (beat_this)  │     │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘     │
│           └──────────────┬────┴──────────────────┘              │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Shared Encoder                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Temporal CNN + Transformer (neighborhood attention)       │ │
│  │  Captures local and long-range dependencies                │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼─────┐  ┌───────▼───────┐  ┌─────▼─────────┐
│ Boundary Head │  │  Energy Head  │  │  Event Head   │
│ (frame-wise)  │  │  (regression) │  │  (optional)   │
│               │  │               │  │               │
│ P(boundary)   │  │ energy: 0-1   │  │ P(drop/kick)  │
│ per frame     │  │ per frame     │  │ per frame     │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Post-Processing                             │
│  - Peak picking for boundaries                                   │
│  - Energy smoothing                                              │
│  - Snap to beat grid                                             │
│  - Minimum section duration filtering                            │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Approach

### Phase 1: Foundation Model Integration

Use pretrained MERT as feature backbone:
- Input: 30s audio chunks at 24kHz
- Output: 768-dim embeddings per ~20ms frame
- Fine-tune last 2-3 transformer layers, freeze rest

```python
# src/edm/models/backbone.py
class MERTBackbone:
    def __init__(self, model_name: str = "m-a-p/MERT-v1-330M"):
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze most layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze last 2 layers
        for layer in self.model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Returns [batch, time, 768]
        return self.model(audio).last_hidden_state
```

### Phase 2: Task-Specific Heads

Separate prediction heads for each task:

```python
# src/edm/models/heads.py
class BoundaryHead(nn.Module):
    """Frame-wise boundary probability."""
    def __init__(self, input_dim: int = 768):
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.out = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, feat] -> [batch, time, 1]
        x = x.transpose(1, 2)  # [batch, feat, time]
        x = F.relu(self.conv(x))
        x = torch.sigmoid(self.out(x))
        return x.transpose(1, 2)

class EnergyHead(nn.Module):
    """Frame-wise energy regression."""
    def __init__(self, input_dim: int = 768, num_bands: int = 3):
        # num_bands: bass, mid, high energy
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.out = nn.Conv1d(64, num_bands, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = torch.sigmoid(self.out(x))  # 0-1 per band
        return x.transpose(1, 2)
```

### Phase 3: Training Pipeline

#### Dataset Class
```python
# src/edm/training/dataset.py
class DJLabelDataset(Dataset):
    """Dataset from DJ software labels."""

    def __init__(
        self,
        annotation_dir: Path,
        audio_dir: Path,
        sample_rate: int = 24000,
        chunk_duration: float = 30.0,
    ):
        self.annotations = self._load_annotations(annotation_dir)
        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]

        # Load random chunk from track
        audio = self._load_random_chunk(ann['file'])

        # Generate frame-level labels
        boundary_labels = self._cue_points_to_boundary_frames(
            ann['cue_points'], ann['bpm'], ann['downbeat']
        )
        energy_labels = self._compute_energy_labels(audio)

        return {
            'audio': audio,
            'boundary': boundary_labels,
            'energy': energy_labels,
            'bpm': ann['bpm'],
            'beat_grid': ann['beat_grid'],
        }
```

#### Boundary-Tolerant Loss
Following beat_this paper - loss tolerant to ±N frames:

```python
# src/edm/training/losses.py
def boundary_tolerant_loss(
    pred: torch.Tensor,  # [batch, time, 1]
    target: torch.Tensor,  # [batch, time, 1]
    tolerance_frames: int = 3,
) -> torch.Tensor:
    """BCE loss where target is smeared within tolerance window."""
    # Dilate target labels within tolerance
    kernel = torch.ones(1, 1, tolerance_frames * 2 + 1, device=target.device)
    target_dilated = F.max_pool1d(
        target.transpose(1, 2),
        kernel_size=tolerance_frames * 2 + 1,
        stride=1,
        padding=tolerance_frames
    ).transpose(1, 2)

    return F.binary_cross_entropy(pred, target_dilated)

def multi_task_loss(
    boundary_pred: torch.Tensor,
    boundary_target: torch.Tensor,
    energy_pred: torch.Tensor,
    energy_target: torch.Tensor,
    weights: dict = {'boundary': 1.0, 'energy': 1.0},
) -> torch.Tensor:
    """Combined multi-task loss."""
    boundary_loss = boundary_tolerant_loss(boundary_pred, boundary_target)
    energy_loss = F.mse_loss(energy_pred, energy_target)

    return weights['boundary'] * boundary_loss + weights['energy'] * energy_loss
```

### Phase 4: Label Error Handling

#### Cleanlab Integration
```python
# scripts/validate_labels.py
from cleanlab.classification import CleanLearning

def find_label_errors(
    model: nn.Module,
    dataset: DJLabelDataset,
) -> list[int]:
    """Identify likely label errors using confident learning."""
    # Get cross-validated predictions
    pred_probs = cross_val_predict_proba(model, dataset)

    # Get noisy labels
    noisy_labels = [d['boundary'] for d in dataset]

    # Find issues
    cl = CleanLearning()
    issues = cl.find_label_issues(noisy_labels, pred_probs)

    return issues['is_label_issue'].tolist()
```

#### Heuristic Pre-filtering
```python
def detect_beat_grid_errors(
    cue_points: list[float],
    beat_times: list[float],
    tolerance_ms: float = 50.0,
) -> list[int]:
    """Flag cue points not snapped to beats."""
    errors = []
    for i, cue in enumerate(cue_points):
        nearest_beat = min(beat_times, key=lambda b: abs(b - cue))
        if abs(cue - nearest_beat) * 1000 > tolerance_ms:
            errors.append(i)
    return errors
```

### Phase 5: Inference Integration

```python
# src/edm/analysis/structure_detector.py
class MLDetector:
    """ML-based structure detection using trained model."""

    def __init__(
        self,
        model_path: Path | None = None,
        device: str | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        # Load audio
        audio = librosa.load(filepath, sr=24000)[0]
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            boundary_probs, energy = self.model(audio_tensor)

        # Post-process
        boundaries = self._pick_boundaries(boundary_probs)

        # Create sections with energy values
        return self._boundaries_to_sections(boundaries, energy, len(audio) / 24000)
```

## Data Pipeline

```
DJ Software Export
       │
       ▼
┌──────────────────┐
│  Parse Labels    │
│  (Rekordbox XML, │
│   Serato, etc)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Heuristic Filter │
│ - Beat snapping  │
│ - Duration check │
│ - Format valid   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Split Tiers      │
│ T1: Verified     │◄─── Manual review
│ T2: Auto-clean   │
│ T3: Noisy        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Training Loop   │
│  (noise-robust)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Cleanlab Check  │───► Label corrections
└──────────────────┘
```

## Model Selection Rationale

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **MERT (chosen)** | Music-specific pretraining, beat/key awareness | 330M params, slower | Best for accuracy |
| wav2vec 2.0 | Smaller, faster | Speech-optimized, worse on music | Backup option |
| Train from scratch | Full control | Needs 10x more data | Not recommended |
| All-In-One | SOTA structure | Complex dependencies, less flexible | Consider later |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MERT doesn't capture EDM patterns | Fine-tune on EDM data, add spectral features |
| Not enough clean data | Semi-supervised + data augmentation |
| Inference too slow | Quantization, ONNX export, smaller backbone |
| Energy labels noisy | Compute from audio directly, not from annotations |
| Model doesn't generalize | Held-out genre testing, dropout, early stopping |
