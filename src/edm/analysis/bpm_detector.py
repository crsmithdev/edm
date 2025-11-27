"""BPM computation using beat_this and librosa."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import structlog

from edm.io.audio import AudioData, load_audio

logger = structlog.get_logger(__name__)


@dataclass
class ComputedBPM:
    """Result of computed BPM detection.

    Attributes:
        bpm: Detected BPM value.
        confidence: Confidence score between 0 and 1.
        method: Detection method used ('beat-this' or 'librosa').
        alternatives: Alternative BPM candidates (for tempo multiplicity).
    """

    bpm: float
    confidence: float
    method: Literal["beat-this", "librosa"]
    alternatives: list[float] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


def compute_bpm_beat_this(filepath: Path, device: str | None = None) -> ComputedBPM:
    """Compute BPM using beat_this neural network beat tracker.

    Args:
        filepath: Path to the audio file.
        device: Device for inference ('cuda', 'cpu', or None for auto-detect).

    Returns:
        Computed BPM result with confidence and alternatives.

    Raises:
        AnalysisError: If BPM computation fails.

    Examples:
        >>> result = compute_bpm_beat_this(Path("track.mp3"))
        >>> print(f"BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")
        BPM: 128.0 (confidence: 0.95)
    """
    logger.debug("computing bpm with beat_this", filepath=str(filepath))

    try:
        import torch
        from beat_this.inference import File2Beats

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.debug("beat_this using device", device=device)

        # Initialize beat tracker
        file2beats = File2Beats(checkpoint_path="final0", device=device)

        # Get beat positions
        beats, _downbeats = file2beats(str(filepath))

        if len(beats) < 2:
            logger.warning("insufficient beats detected", beat_count=len(beats))
            from edm.exceptions import AnalysisError

            raise AnalysisError(f"Insufficient beats detected: {len(beats)}")

        # Calculate BPM from beat intervals
        intervals = np.diff(beats)
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval

        # Calculate confidence based on interval consistency
        std = np.std(intervals)
        confidence = max(0.0, min(1.0, 1.0 - (std / median_interval)))

        # Check for tempo multiplicity
        alternatives = []
        for multiplier in [0.5, 2.0]:
            alt_bpm = bpm * multiplier
            if 40 <= alt_bpm <= 200:
                alternatives.append(alt_bpm)

        # Adjust to EDM range
        bpm = _adjust_bpm_to_edm_range(bpm, alternatives)

        logger.debug("detected bpm", bpm=round(bpm, 1), confidence=round(confidence, 2))
        return ComputedBPM(
            bpm=bpm, confidence=confidence, method="beat-this", alternatives=alternatives
        )

    except ImportError as e:
        logger.error("beat_this not installed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError("beat_this library not installed")
    except Exception as e:
        logger.error("beat_this bpm computation failed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"BPM computation failed: {e}")


def compute_bpm_librosa(
    filepath: Path,
    hop_length: int = 512,
    audio: AudioData | None = None,
) -> ComputedBPM:
    """Compute BPM using librosa's tempo detection.

    Args:
        filepath: Path to the audio file.
        hop_length: Hop length for analysis (default: 512).
        audio: Pre-loaded audio data as (y, sr) tuple. If provided, skips loading from disk.

    Returns:
        Computed BPM result.

    Raises:
        AnalysisError: If BPM computation fails.

    Examples:
        >>> result = compute_bpm_librosa(Path("track.mp3"))
        >>> print(f"BPM: {result.bpm:.1f}")
        BPM: 128.0

        >>> # With pre-loaded audio
        >>> y, sr = load_audio(Path("track.mp3"))
        >>> result = compute_bpm_librosa(Path("track.mp3"), audio=(y, sr))
    """
    logger.debug("computing bpm with librosa", filepath=str(filepath))

    try:
        # Use provided audio or load from cache
        if audio is not None:
            y, sr = audio
        else:
            y, sr = load_audio(filepath)

        # Compute tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

        # librosa returns tempo as numpy array, extract scalar
        bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)

        # Librosa's confidence is not directly available, estimate from beat consistency
        if len(beats) > 1:
            librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            intervals = np.diff(beat_times)
            std = np.std(intervals)
            mean_interval = np.mean(intervals)
            confidence = max(0.0, min(1.0, 1.0 - (std / mean_interval)))
        else:
            confidence = 0.5

        # Check for tempo multiplicity
        alternatives = []
        for multiplier in [0.5, 2.0]:
            alt_bpm = bpm * multiplier
            if 40 <= alt_bpm <= 200:
                alternatives.append(alt_bpm)

        # Adjust to EDM range
        bpm = _adjust_bpm_to_edm_range(bpm, alternatives)

        logger.debug("detected bpm", bpm=round(bpm, 1), confidence=round(confidence, 2))
        return ComputedBPM(
            bpm=bpm, confidence=confidence, method="librosa", alternatives=alternatives
        )

    except Exception as e:
        logger.error("librosa bpm computation failed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"BPM computation failed: {e}")


def _adjust_bpm_to_edm_range(bpm: float, alternatives: list[float]) -> float:
    """Adjust BPM to preferred EDM range (120-150).

    If BPM is outside this range and there's an alternative within it,
    prefer the alternative.

    Args:
        bpm: Primary BPM value.
        alternatives: Alternative BPM candidates.

    Returns:
        Adjusted BPM value.
    """
    # Preferred EDM range
    preferred_min, preferred_max = 120, 150

    # If primary BPM is in range, use it
    if preferred_min <= bpm <= preferred_max:
        return bpm

    # Check if any alternative is in preferred range
    for alt_bpm in alternatives:
        if preferred_min <= alt_bpm <= preferred_max:
            logger.debug(
                "adjusting bpm for edm range preference",
                original_bpm=round(bpm, 1),
                adjusted_bpm=round(alt_bpm, 1),
            )
            return alt_bpm

    # No alternative in range, return original
    return bpm


def compute_bpm(
    filepath: Path,
    prefer_madmom: bool = True,
    librosa_hop_length: int = 512,
    audio: AudioData | None = None,
    device: str | None = None,
) -> ComputedBPM:
    """Compute BPM using available methods.

    Tries beat_this first if preferred and available, falls back to librosa.

    Args:
        filepath: Path to the audio file.
        prefer_madmom: Use neural network (beat_this) if True, librosa if False.
            Parameter name kept for backward compatibility.
        librosa_hop_length: Hop length for librosa (default: 512).
        audio: Pre-loaded audio data as (y, sr) tuple. If provided, skips loading
            from disk (only used by librosa fallback).
        device: Device for beat_this inference ('cuda', 'cpu', or None for auto).

    Returns:
        Computed BPM result.

    Raises:
        AnalysisError: If all computation methods fail.
    """
    if prefer_madmom:
        try:
            # beat_this loads its own audio
            return compute_bpm_beat_this(filepath, device=device)
        except Exception as e:
            logger.warning("beat_this failed, falling back to librosa", error=str(e))
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length, audio=audio)
    else:
        try:
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length, audio=audio)
        except Exception as e:
            logger.warning("librosa failed, trying beat_this", error=str(e))
            return compute_bpm_beat_this(filepath, device=device)
