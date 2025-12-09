"""ML-based structure detection using trained models."""

from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import structlog
import torch

from edm.analysis.structure_detector import DetectedSection
from edm.models.multitask import MultiTaskModel

logger = structlog.get_logger(__name__)


class MLStructureDetector:
    """ML-based structure detection using multi-task model.

    Uses a trained MultiTaskModel to predict:
    - Section boundaries (frame-level classification)
    - Section labels (intro, buildup, drop, breakdown, outro)
    - Energy levels (bass, mid, high)
    - Beat locations

    Post-processes predictions to produce DetectedSection objects.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str | None = None,
        boundary_threshold: float = 0.5,
        peak_distance_seconds: float = 4.0,
        min_section_duration: float = 8.0,
        snap_to_beats: bool = True,
    ):
        """Initialize ML structure detector.

        Args:
            model_path: Path to trained model checkpoint. If None, uses default.
            device: Device for inference ('cuda', 'cpu', or None for auto).
            boundary_threshold: Threshold for boundary detection [0-1].
            peak_distance_seconds: Minimum distance between boundaries in seconds.
            min_section_duration: Minimum section duration in seconds.
            snap_to_beats: Snap boundaries to nearest beats.
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.boundary_threshold = boundary_threshold
        self.peak_distance_seconds = peak_distance_seconds
        self.min_section_duration = min_section_duration
        self.snap_to_beats = snap_to_beats

        # Lazy-load model (only when needed)
        self._model: MultiTaskModel | None = None

    def _load_model(self) -> MultiTaskModel:
        """Load model from checkpoint.

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If no model_path provided and default not found
        """
        if self._model is not None:
            return self._model

        if self.model_path is None:
            # Try to find default model in experiments/
            default_paths = [
                Path("experiments/best_model.pt"),
                Path("models/structure_detector.pt"),
            ]
            for path in default_paths:
                if path.exists():
                    self.model_path = path
                    logger.info("using default model", path=str(path))
                    break

            if self.model_path is None:
                raise RuntimeError(
                    "No model_path provided and no default model found. "
                    "Train a model first or provide explicit path."
                )

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("loading ml model", path=str(model_path), device=self.device)
        self._model = MultiTaskModel.load(model_path, device=self.device)
        self._model.eval()

        return self._model

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        """Detect structure using ML model.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate for audio loading.

        Returns:
            List of detected sections with EDM-specific labels.
        """
        logger.debug("running ml structure detection", filepath=str(filepath))

        # Load model
        model = self._load_model()

        # Load audio
        y, loaded_sr = librosa.load(str(filepath), sr=sr, mono=True)
        actual_sr = int(loaded_sr)
        duration = len(y) / actual_sr

        # Run inference
        with torch.no_grad():
            audio_tensor = torch.from_numpy(y).float().unsqueeze(0).to(self.device)
            predictions = model(audio_tensor)

        # Extract predictions (move to CPU)
        boundary_probs = predictions["boundary"].squeeze().cpu().numpy()  # [frames, 1]
        label_logits = predictions["label"].squeeze().cpu().numpy()  # [frames, num_classes]
        beat_probs = predictions.get("beat", None)

        if beat_probs is not None:
            beat_probs = beat_probs.squeeze().cpu().numpy()  # [frames, 1]

        # Get frame rate from model (should match training)
        # Default to 68.87 Hz (MERT frame rate)
        frame_rate = 68.87
        if hasattr(model.backbone, "get_frame_rate") and callable(model.backbone.get_frame_rate):
            frame_rate = float(model.backbone.get_frame_rate())

        # Post-process predictions
        sections = self._predictions_to_sections(
            boundary_probs=boundary_probs.squeeze(),  # Remove channel dim
            label_logits=label_logits,
            beat_probs=beat_probs.squeeze() if beat_probs is not None else None,
            frame_rate=frame_rate,
            duration=duration,
        )

        logger.debug("ml detection complete", sections=len(sections))

        return sections

    def _predictions_to_sections(
        self,
        boundary_probs: np.ndarray,
        label_logits: np.ndarray,
        beat_probs: np.ndarray | None,
        frame_rate: float,
        duration: float,
    ) -> list[DetectedSection]:
        """Convert model predictions to DetectedSection objects.

        Args:
            boundary_probs: Boundary probabilities [frames]
            label_logits: Label logits [frames, num_classes]
            beat_probs: Beat probabilities [frames] or None
            frame_rate: Frames per second
            duration: Audio duration in seconds

        Returns:
            List of detected sections
        """
        # Convert label logits to class predictions
        label_probs = np.exp(label_logits) / np.sum(np.exp(label_logits), axis=-1, keepdims=True)
        label_ids = np.argmax(label_logits, axis=-1)

        # Label mapping (must match training)
        id_to_label = {
            0: "intro",
            1: "buildup",
            2: "drop",
            3: "breakdown",
            4: "outro",
        }

        # Detect boundary frames using peak picking
        boundary_frames = self._detect_boundary_peaks(
            boundary_probs,
            frame_rate=frame_rate,
        )

        # Convert frames to times
        boundary_times = boundary_frames / frame_rate

        # Add start and end if not present
        if len(boundary_times) == 0 or boundary_times[0] > 0:
            boundary_times = np.concatenate([[0.0], boundary_times])
        if boundary_times[-1] < duration - 0.1:
            boundary_times = np.concatenate([boundary_times, [duration]])

        # Snap to beats if available and enabled
        if self.snap_to_beats and beat_probs is not None:
            boundary_times = self._snap_to_beats(boundary_times, beat_probs, frame_rate)

        # Create sections from boundaries
        sections = []
        for i in range(len(boundary_times) - 1):
            start_time = float(boundary_times[i])
            end_time = float(boundary_times[i + 1])

            # Get dominant label for this section
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)
            section_labels = label_ids[start_frame:end_frame]

            if len(section_labels) > 0:
                # Most common label in section
                label_id = int(np.bincount(section_labels).argmax())
                label = id_to_label.get(label_id, "unknown")

                # Average confidence for this label
                section_probs = label_probs[start_frame:end_frame, label_id]
                confidence = float(np.mean(section_probs))
            else:
                label = "unknown"
                confidence = 0.5

            sections.append(
                DetectedSection(
                    start_time=start_time,
                    end_time=end_time,
                    label=label,
                    confidence=confidence,
                    is_event=False,
                )
            )

        # Filter out very short sections
        sections = self._merge_short_sections(sections)

        return sections

    def _detect_boundary_peaks(
        self,
        boundary_probs: np.ndarray,
        frame_rate: float,
    ) -> np.ndarray:
        """Detect boundary peaks from probability curve.

        Args:
            boundary_probs: Boundary probabilities [frames]
            frame_rate: Frames per second

        Returns:
            Array of boundary frame indices
        """
        # Convert distance in seconds to frames
        min_distance_frames = int(self.peak_distance_seconds * frame_rate)

        # Find peaks above threshold
        peaks, _ = scipy.signal.find_peaks(
            boundary_probs,
            height=self.boundary_threshold,
            distance=min_distance_frames,
        )

        return np.asarray(peaks)

    def _snap_to_beats(
        self,
        boundary_times: np.ndarray,
        beat_probs: np.ndarray,
        frame_rate: float,
        snap_window_seconds: float = 0.5,
    ) -> np.ndarray:
        """Snap boundary times to nearest detected beats.

        Args:
            boundary_times: Boundary times in seconds
            beat_probs: Beat probabilities [frames]
            frame_rate: Frames per second
            snap_window_seconds: Maximum distance to snap (seconds)

        Returns:
            Snapped boundary times
        """
        # Detect beat frames
        beat_frames, _ = scipy.signal.find_peaks(
            beat_probs,
            height=0.3,
            distance=int(frame_rate * 0.3),  # Min 0.3s between beats (~200 BPM max)
        )
        beat_times = beat_frames / frame_rate

        # Snap each boundary to nearest beat within window
        snapped_times = []
        snap_window_frames = snap_window_seconds * frame_rate

        for boundary_time in boundary_times:
            boundary_frame = boundary_time * frame_rate

            # Find nearest beat within window
            distances = np.abs(beat_frames - boundary_frame)
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]

            if min_distance <= snap_window_frames:
                # Snap to beat
                snapped_times.append(float(beat_times[nearest_idx]))
            else:
                # Keep original
                snapped_times.append(float(boundary_time))

        return np.array(snapped_times)

    def _merge_short_sections(self, sections: list[DetectedSection]) -> list[DetectedSection]:
        """Merge sections shorter than minimum duration.

        Args:
            sections: List of sections

        Returns:
            List with short sections merged into neighbors
        """
        if len(sections) <= 1:
            return sections

        merged = [sections[0]]

        for section in sections[1:]:
            duration = section.end_time - section.start_time
            prev = merged[-1]

            if duration < self.min_section_duration:
                # Merge with previous section
                # Keep label with higher confidence
                if section.confidence > prev.confidence:
                    label = section.label
                    confidence = section.confidence
                else:
                    label = prev.label
                    confidence = prev.confidence

                merged[-1] = DetectedSection(
                    start_time=prev.start_time,
                    end_time=section.end_time,
                    label=label,
                    confidence=confidence,
                    is_event=False,
                )
            else:
                merged.append(section)

        return merged
