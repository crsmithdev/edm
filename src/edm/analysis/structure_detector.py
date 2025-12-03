"""Structure detection implementations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
import scipy
import structlog

# Patch scipy.inf for compatibility with msaf (scipy 1.12+ removed scipy.inf)
# Must be done before importing msaf
if not hasattr(scipy, "inf"):
    scipy.inf = np.inf

import msaf  # noqa: E402

logger = structlog.get_logger(__name__)


@dataclass
class DetectedSection:
    """A detected section from a structure detector.

    Attributes:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        label: Section label (generic or EDM-specific).
        confidence: Confidence score between 0 and 1.
        is_event: True for moment-based events (drops, kicks), False for spans.
        start_bar: Start bar position (0-indexed). None if BPM unavailable.
        end_bar: End bar position. None if BPM unavailable.
        bar_count: Number of bars in section. None if BPM unavailable.
    """

    start_time: float
    end_time: float
    label: str
    confidence: float
    is_event: bool = False
    start_bar: float | None = None
    end_bar: float | None = None
    bar_count: float | None = None


class StructureDetector(Protocol):
    """Protocol for structure detection implementations."""

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        """Detect structure sections in an audio file.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate for audio loading.

        Returns:
            List of detected sections.
        """
        ...


class MSAFDetector:
    """MSAF-based structure detection.

    Uses the Music Structure Analysis Framework for boundary detection
    and segment labeling, with energy-based EDM label mapping.
    """

    def __init__(self, boundary_algorithm: str = "sf", label_algorithm: str = "fmc2d"):
        """Initialize MSAF detector.

        Args:
            boundary_algorithm: MSAF boundary detection algorithm.
                Options: 'sf' (spectral flux), 'foote', 'olda', 'scluster', 'cnmf'.
            label_algorithm: MSAF labeling algorithm.
                Options: 'fmc2d', 'scluster', 'cnmf'.
        """
        self._boundary_algorithm = boundary_algorithm
        self._label_algorithm = label_algorithm

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        """Detect structure using MSAF.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate for audio loading.

        Returns:
            List of detected sections with EDM labels.
        """
        logger.debug(
            "running msaf detection",
            filepath=str(filepath),
            boundary_algorithm=self._boundary_algorithm,
            label_algorithm=self._label_algorithm,
        )

        try:
            # Run MSAF segmentation
            boundaries, labels = msaf.process(
                str(filepath),
                boundaries_id=self._boundary_algorithm,
                labels_id=self._label_algorithm,
            )

            # Get audio duration for the last segment
            y, _ = librosa.load(str(filepath), sr=sr, mono=True)
            duration = len(y) / sr

            # Convert to sections with MSAF cluster labels
            sections = self._boundaries_to_sections(boundaries, labels, duration)

            logger.debug(
                "msaf detection complete",
                sections=len(sections),
            )

            return sections

        except Exception as e:
            logger.error("msaf detection failed", error=str(e))
            raise

    def _boundaries_to_sections(
        self, boundaries: np.ndarray, labels: np.ndarray, duration: float
    ) -> list[DetectedSection]:
        """Convert MSAF boundaries and labels to sections.

        Args:
            boundaries: Array of boundary times.
            labels: Array of segment cluster IDs from MSAF.
            duration: Total audio duration.

        Returns:
            List of detected sections with cluster-based labels (e.g., "segment1").
        """
        sections = []

        for i in range(len(boundaries) - 1):
            start = float(boundaries[i])
            end = float(boundaries[i + 1])
            # Use 1-indexed segment labels based on MSAF cluster ID
            cluster_id = int(labels[i]) + 1 if i < len(labels) else 0
            label = f"segment{cluster_id}"

            sections.append(
                DetectedSection(
                    start_time=start,
                    end_time=end,
                    label=label,
                    confidence=0.8,  # MSAF doesn't provide confidence scores
                )
            )

        # Ensure last section extends to duration
        if sections and sections[-1].end_time < duration:
            sections[-1] = DetectedSection(
                start_time=sections[-1].start_time,
                end_time=duration,
                label=sections[-1].label,
                confidence=sections[-1].confidence,
            )

        return sections


def merge_short_sections(
    sections: list[DetectedSection],
    bpm: float | None = None,
    min_section_bars: int = 8,
    time_signature: tuple[int, int] = (4, 4),
) -> list[DetectedSection]:
    """Merge sections shorter than minimum bar count.

    Args:
        sections: List of sections to process.
        bpm: BPM for bar calculation. If None, uses a default 8-second minimum.
        min_section_bars: Minimum section length in bars. Default 8.
        time_signature: Time signature tuple. Default (4, 4).

    Returns:
        List with short sections merged into adjacent sections.
    """
    if len(sections) <= 1:
        return sections

    # Calculate minimum duration in seconds
    # At typical EDM tempos (120-150 BPM), 8 bars = 16-13 seconds
    # Use 8 seconds as fallback if no BPM (conservative estimate for ~150 BPM)
    beats_per_bar = time_signature[0]
    if bpm and bpm > 0:
        seconds_per_beat = 60.0 / bpm
        min_duration = min_section_bars * beats_per_bar * seconds_per_beat
    else:
        min_duration = 8.0  # Fallback: 8 seconds

    logger.debug(
        "merging short sections",
        min_bars=min_section_bars,
        min_duration_seconds=round(min_duration, 2),
        input_sections=len(sections),
    )

    merged: list[DetectedSection] = []

    for section in sections:
        duration = section.end_time - section.start_time

        if not merged:
            merged.append(section)
            continue

        prev = merged[-1]

        if duration < min_duration:
            # Merge with previous section (extend previous to cover this one)
            merged[-1] = DetectedSection(
                start_time=prev.start_time,
                end_time=section.end_time,
                label=prev.label,  # Keep previous label
                confidence=max(prev.confidence, section.confidence),
                is_event=prev.is_event,
            )
        else:
            merged.append(section)

    logger.debug("merged short sections", output_sections=len(merged))

    return merged


class EnergyDetector:
    """Energy-based structure detection using librosa.

    Uses RMS energy, spectral contrast, and onset strength
    for rule-based drop/breakdown/buildup detection.
    """

    def __init__(
        self,
        min_section_duration: float = 8.0,
        energy_threshold_high: float = 0.7,
        energy_threshold_low: float = 0.4,
    ):
        """Initialize energy detector.

        Args:
            min_section_duration: Minimum section duration in seconds.
            energy_threshold_high: Normalized energy threshold for drops.
            energy_threshold_low: Normalized energy threshold for breakdowns.
        """
        self._min_section_duration = min_section_duration
        self._energy_threshold_high = energy_threshold_high
        self._energy_threshold_low = energy_threshold_low

    def detect(self, filepath: Path, sr: int = 22050) -> list[DetectedSection]:
        """Detect structure using energy analysis.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate for audio loading.

        Returns:
            List of detected sections with EDM labels.
        """
        logger.debug("running energy-based detection", filepath=str(filepath))

        # Load audio
        y, loaded_sr = librosa.load(str(filepath), sr=sr, mono=True)
        actual_sr: int = int(loaded_sr)
        duration = len(y) / actual_sr

        # Calculate frame-level RMS energy
        hop_length: int = 512
        frame_length: int = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Smooth RMS with median filter
        from scipy.ndimage import median_filter

        rms_smooth = median_filter(rms, size=21)

        # Normalize
        rms_norm = rms_smooth / (np.max(rms_smooth) + 1e-8)

        # Detect boundaries using energy changes
        boundaries = self._detect_boundaries(rms_norm, hop_length, actual_sr, duration)

        # Create sections from boundaries
        sections = self._boundaries_to_sections(
            boundaries, rms_norm, hop_length, actual_sr, duration
        )

        # Merge short sections
        sections = self._merge_short_sections(sections)

        logger.debug("energy detection complete", sections=len(sections))

        return sections

    def _detect_boundaries(
        self, rms_norm: np.ndarray, hop_length: int, sr: int, duration: float
    ) -> list[float]:
        """Detect section boundaries from energy changes.

        Args:
            rms_norm: Normalized RMS energy.
            hop_length: Hop length in samples.
            sr: Sample rate.
            duration: Total duration.

        Returns:
            List of boundary times in seconds.
        """
        # Calculate energy gradient
        gradient = np.gradient(rms_norm)

        # Find significant changes (peaks in absolute gradient)
        from scipy.signal import find_peaks

        # Find positive peaks (energy increases - potential drop starts)
        pos_peaks, _ = find_peaks(gradient, height=0.02, distance=sr // hop_length * 4)

        # Find negative peaks (energy decreases - potential breakdown starts)
        neg_peaks, _ = find_peaks(-gradient, height=0.02, distance=sr // hop_length * 4)

        # Combine and sort boundaries
        all_peaks = np.concatenate([pos_peaks, neg_peaks])
        all_peaks = np.unique(np.sort(all_peaks))

        # Convert to times
        frame_times = librosa.frames_to_time(all_peaks, sr=sr, hop_length=hop_length)

        # Add start and end (convert numpy floats to Python floats)
        boundaries: list[float] = [0.0] + [float(t) for t in frame_times] + [float(duration)]

        # Filter out boundaries that are too close
        filtered: list[float] = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= self._min_section_duration:
                filtered.append(b)

        # Ensure we have end boundary
        if filtered[-1] < duration - 1.0:
            filtered.append(float(duration))

        return filtered

    def _boundaries_to_sections(
        self,
        boundaries: list[float],
        rms_norm: np.ndarray,
        hop_length: int,
        sr: int,
        duration: float,
    ) -> list[DetectedSection]:
        """Convert boundaries to labeled sections.

        Args:
            boundaries: List of boundary times.
            rms_norm: Normalized RMS energy.
            hop_length: Hop length in samples.
            sr: Sample rate.
            duration: Total duration.

        Returns:
            List of detected sections.
        """
        sections = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Calculate average energy for this section
            start_frame = int(start * sr / hop_length)
            end_frame = int(end * sr / hop_length)
            end_frame = min(end_frame, len(rms_norm))

            if start_frame < end_frame:
                avg_energy = np.mean(rms_norm[start_frame:end_frame])
            else:
                avg_energy = 0.5

            # Determine label and whether it's an event
            is_event = False
            if i == 0:
                label = "intro"
                confidence = 0.9
            elif i == len(boundaries) - 2:
                label = "outro"
                confidence = 0.9
            elif avg_energy > self._energy_threshold_high:
                label = "drop"
                confidence = 0.8 + float(avg_energy - self._energy_threshold_high) * 0.5
                is_event = True  # Drops are events
            elif avg_energy < self._energy_threshold_low:
                label = "breakdown"
                confidence = 0.75
            else:
                # Mid-energy section - doesn't fit known EDM categories
                label = "other"
                confidence = 0.5

            confidence = float(min(confidence, 0.99))

            sections.append(
                DetectedSection(
                    start_time=float(start),
                    end_time=float(end),
                    label=label,
                    confidence=confidence,
                    is_event=is_event,
                )
            )

        return sections

    def _merge_short_sections(self, sections: list[DetectedSection]) -> list[DetectedSection]:
        """Merge sections shorter than minimum duration.

        Args:
            sections: List of sections.

        Returns:
            List with short sections merged.
        """
        if len(sections) <= 1:
            return sections

        merged = [sections[0]]

        for section in sections[1:]:
            duration = section.end_time - section.start_time
            prev = merged[-1]
            prev_duration = prev.end_time - prev.start_time

            if duration < self._min_section_duration:
                # Merge with previous section
                merged[-1] = DetectedSection(
                    start_time=float(prev.start_time),
                    end_time=float(section.end_time),
                    label=prev.label if prev_duration >= duration else section.label,
                    confidence=float(max(prev.confidence, section.confidence)),
                )
            else:
                merged.append(section)

        return merged


def get_detector(detector_type: str) -> StructureDetector:
    """Get a structure detector by type.

    Args:
        detector_type: Detector type ('msaf', 'energy').

    Returns:
        Detector instance.

    Raises:
        ValueError: If detector_type is unknown.
    """
    if detector_type == "energy":
        return EnergyDetector()

    if detector_type in ("msaf", "auto"):
        # 'auto' is now an alias for 'msaf' (msaf is required)
        return MSAFDetector()

    raise ValueError(f"Unknown detector type: {detector_type}")
