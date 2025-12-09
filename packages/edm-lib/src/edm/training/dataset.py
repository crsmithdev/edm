"""PyTorch Dataset for EDM structure detection training."""

from pathlib import Path
from typing import Any
from urllib.parse import unquote

import librosa
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from edm.data.schema import Annotation


class EDMDataset(Dataset):
    """Dataset for EDM structure detection.

    Loads audio files and their annotations for multi-task learning:
    - Boundary detection (frame-level binary classification)
    - Energy prediction (frame-level regression, 3 bands)
    - Beat detection (frame-level binary classification)
    - Label classification (frame-level multi-class, optional)

    Attributes:
        annotation_dir: Directory containing YAML annotations
        audio_dir: Root directory for audio files
        sample_rate: Audio sample rate
        duration: Fixed duration to load (in seconds, None = full track)
        augment: Enable audio augmentation
    """

    def __init__(
        self,
        annotation_dir: Path,
        audio_dir: Path | None = None,
        sample_rate: int = 22050,
        duration: float | None = 30.0,
        augment: bool = False,
        frame_rate: float = 68.87,  # Match MERT output frame rate (~320 samples per frame)
        tier_filter: int | None = None,
        min_confidence: float | None = None,
    ):
        """Initialize dataset.

        Args:
            annotation_dir: Directory with YAML annotation files
            audio_dir: Root directory for audio files (if None, use paths from annotations)
            sample_rate: Target sample rate
            duration: Segment duration in seconds (None = full track)
            augment: Enable data augmentation
            frame_rate: Target frame rate for labels (frames per second)
            tier_filter: Only include annotations of this tier (1, 2, or 3). None = all tiers
            min_confidence: Only include annotations with confidence >= this value. None = all
        """
        self.annotation_dir = Path(annotation_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        self.frame_rate = frame_rate
        self.tier_filter = tier_filter
        self.min_confidence = min_confidence

        # Load all annotation files with preference for reference (tier 1) over generated (tier 2)
        annotation_map: dict[str, tuple[Path, Annotation, int]] = {}
        skipped_files = 0

        for yaml_path in sorted(self.annotation_dir.rglob("*.yaml")):
            try:
                with open(yaml_path) as f:
                    # Load first document (handles files with multiple YAML docs)
                    docs = list(yaml.safe_load_all(f))
                    data = docs[0] if docs else {}
                annotation = Annotation(**data)

                # Track key by audio filename stem
                audio_stem = Path(annotation.audio.file).stem
                tier = annotation.metadata.tier

                # Apply filters
                if tier_filter is not None and tier != tier_filter:
                    continue
                if min_confidence is not None and annotation.metadata.confidence < min_confidence:
                    continue

                # Validate audio file exists
                audio_path = self._resolve_audio_path(annotation.audio.file)
                if not audio_path.exists():
                    skipped_files += 1
                    continue

                # Prefer lower tier number (1 = reference, best quality)
                if audio_stem not in annotation_map or tier < annotation_map[audio_stem][2]:
                    annotation_map[audio_stem] = (yaml_path, annotation, tier)

            except Exception as e:
                print(f"Warning: Failed to load {yaml_path}: {e}")

        # Use deduplicated annotations
        self.annotations: list[tuple[Path, Annotation]] = [
            (path, ann) for path, ann, _ in annotation_map.values()
        ]

        print(f"Loaded {len(self.annotations)} annotations from {annotation_dir}")
        if skipped_files > 0:
            print(f"Skipped {skipped_files} annotations with missing audio files")

        # Build label vocabulary from all annotations
        self.label_vocab = self._build_label_vocabulary()
        self.num_classes = len(self.label_vocab)
        print(f"Detected {self.num_classes} unique labels: {sorted(self.label_vocab.keys())}")

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dict with:
                - audio: Audio waveform [samples]
                - boundary: Boundary targets [frames, 1]
                - energy: Energy targets [frames, 3]
                - beat: Beat targets [frames, 1]
                - label: Label targets [frames] (optional)
                - bpm: BPM (float)
                - duration: Actual duration (float)
        """
        yaml_path, annotation = self.annotations[idx]

        # Resolve audio path
        audio_path = self._resolve_audio_path(annotation.audio.file)

        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.duration,
            mono=True,
        )

        # Get actual duration
        actual_duration = len(audio) / self.sample_rate

        # Calculate number of frames
        num_frames = int(actual_duration * self.frame_rate)

        # Create frame-level targets
        targets = self._create_targets(annotation, num_frames, actual_duration)

        # Extract confidence from annotation metadata
        confidence_score = (
            annotation.metadata.confidence if hasattr(annotation, "metadata") else 1.0
        )

        return {
            "audio": torch.from_numpy(audio).float(),
            "boundary": torch.from_numpy(targets["boundary"]).float(),
            "energy": torch.from_numpy(targets["energy"]).float(),
            "beat": torch.from_numpy(targets["beat"]).float(),
            "label": torch.from_numpy(targets["label"]).long(),
            "confidence": torch.tensor(confidence_score, dtype=torch.float32),
            "bpm": annotation.audio.bpm,
            "duration": actual_duration,
        }

    def _build_label_vocabulary(self) -> dict[str, int]:
        """Build label vocabulary from all loaded annotations.

        Returns:
            Dict mapping label names to integer indices
        """
        unique_labels = set()
        for _, annotation in self.annotations:
            for section in annotation.structure:
                unique_labels.add(section.label)

        # Sort labels for consistent ordering
        sorted_labels = sorted(unique_labels)
        return {label: idx for idx, label in enumerate(sorted_labels)}

    def _resolve_audio_path(self, annotation_path: Path) -> Path:
        """Resolve audio file path from annotation.

        Args:
            annotation_path: Path from annotation file

        Returns:
            Resolved absolute path
        """
        if self.audio_dir:
            # Use audio_dir + filename (URL-decode the filename)
            filename = unquote(annotation_path.name)
            return self.audio_dir / filename
        else:
            # Use path from annotation (might need fixing for cross-platform)
            path_str = str(annotation_path)
            # Handle Windows paths like /C:/Music/...
            if path_str.startswith("/") and ":" in path_str:
                path_str = path_str[1:]  # Remove leading /
            # URL-decode the path
            path_str = unquote(path_str)
            return Path(path_str)

    def _create_targets(
        self, annotation: Annotation, num_frames: int, duration: float
    ) -> dict[str, np.ndarray]:
        """Create frame-level targets from annotations.

        Args:
            annotation: Annotation object
            num_frames: Number of frames
            duration: Audio duration

        Returns:
            Dict with target arrays
        """
        # Initialize targets
        boundary = np.zeros((num_frames, 1), dtype=np.float32)
        energy = np.zeros((num_frames, 3), dtype=np.float32)
        beat = np.zeros((num_frames, 1), dtype=np.float32)
        label = np.zeros(num_frames, dtype=np.int64)

        # Use learned label vocabulary
        label_map = self.label_vocab
        default_label = label_map.get("unlabeled", 0)  # fallback to first label if no 'unlabeled'

        # Process structure sections
        for i, section in enumerate(annotation.structure):
            # Get frame index for this timestamp
            frame_idx = int(section.time * self.frame_rate)
            if frame_idx >= num_frames:
                continue

            # Mark boundary (with Gaussian kernel for soft targets)
            boundary = self._mark_boundary(boundary, frame_idx, sigma=2)

            # Get section end time
            if i + 1 < len(annotation.structure):
                section_end = annotation.structure[i + 1].time
            else:
                section_end = duration

            # Fill section label (unknown labels map to 'unlabeled')
            section_label = label_map.get(section.label, default_label)
            start_frame = frame_idx
            end_frame = min(int(section_end * self.frame_rate), num_frames)
            label[start_frame:end_frame] = section_label

            # Energy estimation (simplified - could be enhanced with actual audio analysis)
            section_energy = self._estimate_energy(section.label)
            energy[start_frame:end_frame] = section_energy

        # Generate beat targets from BPM
        beat = self._generate_beat_targets(
            annotation.audio.bpm,
            annotation.audio.downbeat,
            num_frames,
            duration,
        )

        return {
            "boundary": boundary,
            "energy": energy,
            "beat": beat,
            "label": label,
        }

    def _mark_boundary(
        self, boundary: np.ndarray, frame_idx: int, sigma: float = 2.0
    ) -> np.ndarray:
        """Mark boundary with Gaussian kernel.

        Args:
            boundary: Boundary array [frames, 1]
            frame_idx: Frame index to mark
            sigma: Gaussian standard deviation

        Returns:
            Updated boundary array
        """
        num_frames = len(boundary)
        # Create Gaussian kernel
        kernel_size = int(sigma * 6)  # 3 sigma on each side
        x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))

        # Apply kernel
        for i, val in enumerate(kernel):
            idx = frame_idx + x[i]
            if 0 <= idx < num_frames:
                boundary[idx, 0] = max(boundary[idx, 0], val)

        return boundary

    def _estimate_energy(self, label: str) -> np.ndarray:
        """Estimate energy levels from section label.

        This is a simplified heuristic. In a real system, you'd compute
        actual energy from the audio.

        Args:
            label: Section label

        Returns:
            Energy array [3] for bass/mid/high
        """
        energy_map = {
            "intro": np.array([0.3, 0.3, 0.2], dtype=np.float32),
            "buildup": np.array([0.5, 0.6, 0.7], dtype=np.float32),
            "drop": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "breakdown": np.array([0.4, 0.5, 0.3], dtype=np.float32),
            "breakbuild": np.array([0.5, 0.6, 0.6], dtype=np.float32),
            "outro": np.array([0.2, 0.3, 0.2], dtype=np.float32),
        }
        return energy_map.get(label, np.array([0.5, 0.5, 0.5], dtype=np.float32))

    def _generate_beat_targets(
        self, bpm: float, downbeat: float, num_frames: int, duration: float
    ) -> np.ndarray:
        """Generate beat targets from BPM and downbeat.

        Args:
            bpm: Beats per minute
            downbeat: First downbeat time
            num_frames: Number of frames
            duration: Audio duration

        Returns:
            Beat targets [frames, 1]
        """
        beat = np.zeros((num_frames, 1), dtype=np.float32)

        # Calculate beat interval
        beat_interval = 60.0 / bpm

        # Generate beat times
        beat_time = downbeat
        while beat_time < duration:
            frame_idx = int(beat_time * self.frame_rate)
            if frame_idx < num_frames:
                # Mark beat with Gaussian
                beat = self._mark_boundary(beat, frame_idx, sigma=1.0)
            beat_time += beat_interval

        return beat


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for DataLoader.

    Handles variable-length sequences by padding to max length in batch.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dict with padded tensors
    """
    # Find max audio length
    max_audio_len = max(item["audio"].shape[0] for item in batch)
    max_frames = max(item["boundary"].shape[0] for item in batch)

    # Initialize batched tensors
    batch_size = len(batch)
    audio_batch = torch.zeros(batch_size, max_audio_len)
    boundary_batch = torch.zeros(batch_size, max_frames, 1)
    energy_batch = torch.zeros(batch_size, max_frames, 3)
    beat_batch = torch.zeros(batch_size, max_frames, 1)
    label_batch = torch.zeros(batch_size, max_frames, dtype=torch.long)

    bpm_batch = []
    duration_batch = []
    confidence_batch = []

    # Fill batches
    for i, item in enumerate(batch):
        audio_len = item["audio"].shape[0]
        num_frames = item["boundary"].shape[0]

        audio_batch[i, :audio_len] = item["audio"]
        boundary_batch[i, :num_frames] = item["boundary"]
        energy_batch[i, :num_frames] = item["energy"]
        beat_batch[i, :num_frames] = item["beat"]
        label_batch[i, :num_frames] = item["label"]

        bpm_batch.append(item["bpm"])
        duration_batch.append(item["duration"])
        confidence_batch.append(item["confidence"])

    return {
        "audio": audio_batch,
        "boundary": boundary_batch,
        "energy": energy_batch,
        "beat": beat_batch,
        "label": label_batch,
        "confidence": torch.stack(confidence_batch),
        "bpm": torch.tensor(bpm_batch),
        "duration": torch.tensor(duration_batch),
    }


def create_dataloaders(
    annotation_dir: Path,
    audio_dir: Path | None = None,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 4,
    **dataset_kwargs: Any,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        annotation_dir: Directory with annotations
        audio_dir: Audio files directory
        batch_size: Batch size
        train_split: Fraction for training (rest is validation)
        num_workers: Number of dataloader workers
        **dataset_kwargs: Additional arguments for EDMDataset

    Returns:
        (train_loader, val_loader)
    """
    # Create full dataset
    dataset = EDMDataset(
        annotation_dir=annotation_dir,
        audio_dir=audio_dir,
        **dataset_kwargs,
    )

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
