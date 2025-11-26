"""BPM computation using madmom and librosa."""

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
        method: Detection method used ('madmom-dbn' or 'librosa').
        alternatives: Alternative BPM candidates (for tempo multiplicity).
    """

    bpm: float
    confidence: float
    method: Literal["madmom-dbn", "librosa"]
    alternatives: list[float] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


def compute_bpm_madmom(filepath: Path, fps: int = 100) -> ComputedBPM:
    """Compute BPM using madmom's DBN beat tracker.

    Args:
        filepath: Path to the audio file.
        fps: Frames per second for analysis (default: 100).

    Returns:
        Computed BPM result with confidence and alternatives.

    Raises:
        AnalysisError: If BPM computation fails.

    Examples:
        >>> result = compute_bpm_madmom(Path("track.mp3"))
        >>> print(f"BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")
        BPM: 128.0 (confidence: 0.95)
    """
    logger.info("computing bpm with madmom", filepath=str(filepath))

    try:
        import madmom

        # Use RNNBeatProcessor and DBNBeatTrackingProcessor
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=fps)
        act = madmom.features.beats.RNNBeatProcessor()(str(filepath))
        beats = proc(act)

        if len(beats) < 2:
            logger.warning(
                "insufficient beats detected, trying tempo estimator", beat_count=len(beats)
            )
            # Fallback to tempo estimator
            tempo_estimator = madmom.features.tempo.TempoEstimationProcessor(fps=fps)
            tempo_act = madmom.features.tempo.RNNTempoProcessor()(str(filepath))
            tempos = tempo_estimator(tempo_act)

            if len(tempos) > 0:
                bpm = float(tempos[0][0])
                confidence = float(tempos[0][1])
                alternatives = [float(t[0]) for t in tempos[1:3]] if len(tempos) > 1 else []

                # Validate BPM range
                bpm = _adjust_bpm_to_edm_range(bpm, alternatives)

                logger.info("detected bpm", bpm=round(bpm, 1), confidence=round(confidence, 2))
                return ComputedBPM(
                    bpm=bpm, confidence=confidence, method="madmom-dbn", alternatives=alternatives
                )
            else:
                raise ValueError("No tempo detected")

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

        logger.info("detected bpm", bpm=round(bpm, 1), confidence=round(confidence, 2))
        return ComputedBPM(
            bpm=bpm, confidence=confidence, method="madmom-dbn", alternatives=alternatives
        )

    except ImportError:
        logger.error("madmom not installed, cannot compute bpm")
        from edm.exceptions import AnalysisError

        raise AnalysisError("madmom library not installed")
    except Exception as e:
        logger.error("madmom bpm computation failed", error=str(e))
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
    logger.info("computing bpm with librosa", filepath=str(filepath))

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

        logger.info("detected bpm", bpm=round(bpm, 1), confidence=round(confidence, 2))
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
    madmom_fps: int = 100,
    librosa_hop_length: int = 512,
    audio: AudioData | None = None,
) -> ComputedBPM:
    """Compute BPM using available methods.

    Tries madmom first if preferred and available, falls back to librosa.

    Args:
        filepath: Path to the audio file.
        prefer_madmom: Try madmom first if True (default: True).
        madmom_fps: Frames per second for madmom (default: 100).
        librosa_hop_length: Hop length for librosa (default: 512).
        audio: Pre-loaded audio data as (y, sr) tuple. If provided, skips loading from disk.

    Returns:
        Computed BPM result.

    Raises:
        AnalysisError: If all computation methods fail.
    """
    if prefer_madmom:
        try:
            # madmom loads its own audio, doesn't support pre-loaded
            return compute_bpm_madmom(filepath, fps=madmom_fps)
        except Exception as e:
            logger.warning("madmom failed, falling back to librosa", error=str(e))
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length, audio=audio)
    else:
        try:
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length, audio=audio)
        except Exception as e:
            logger.warning("librosa failed, trying madmom", error=str(e))
            return compute_bpm_madmom(filepath, fps=madmom_fps)
