"""BPM computation using madmom and librosa."""

import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

logger = structlog.get_logger(__name__)


@dataclass
class ComputedBPM:
    """Result of computed BPM detection.

    Attributes
    ----------
    bpm : float
        Detected BPM value.
    confidence : float
        Confidence score between 0 and 1.
    method : str
        Detection method used ('madmom-dbn' or 'librosa').
    alternatives : list[float]
        Alternative BPM candidates (for tempo multiplicity).
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

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    fps : int, optional
        Frames per second for analysis (default: 100).

    Returns
    -------
    ComputedBPM
        Computed BPM result with confidence and alternatives.

    Raises
    ------
    AnalysisError
        If BPM computation fails.

    Examples
    --------
    >>> result = compute_bpm_madmom(Path("track.mp3"))
    >>> print(f"BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")
    BPM: 128.0 (confidence: 0.95)
    """
    logger.info(f"Computing BPM with madmom for {filepath}")

    try:
        import madmom

        # Use RNNBeatProcessor and DBNBeatTrackingProcessor
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=fps)
        act = madmom.features.beats.RNNBeatProcessor()(str(filepath))
        beats = proc(act)

        if len(beats) < 2:
            logger.warning(f"Insufficient beats detected ({len(beats)}), trying tempo estimator")
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

                logger.info(f"Detected BPM: {bpm:.1f} (confidence: {confidence:.2f})")
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

        logger.info(f"Detected BPM: {bpm:.1f} (confidence: {confidence:.2f})")
        return ComputedBPM(
            bpm=bpm, confidence=confidence, method="madmom-dbn", alternatives=alternatives
        )

    except ImportError:
        logger.error("madmom not installed, cannot compute BPM")
        from edm.exceptions import AnalysisError

        raise AnalysisError("madmom library not installed")
    except Exception as e:
        logger.error(f"madmom BPM computation failed: {e}")
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"BPM computation failed: {e}")


def compute_bpm_librosa(filepath: Path, hop_length: int = 512) -> ComputedBPM:
    """Compute BPM using librosa's tempo detection.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    hop_length : int, optional
        Hop length for analysis (default: 512).

    Returns
    -------
    ComputedBPM
        Computed BPM result.

    Raises
    ------
    AnalysisError
        If BPM computation fails.

    Examples
    --------
    >>> result = compute_bpm_librosa(Path("track.mp3"))
    >>> print(f"BPM: {result.bpm:.1f}")
    BPM: 128.0
    """
    logger.info(f"Computing BPM with librosa for {filepath}")

    try:
        # Load audio
        y, sr = librosa.load(str(filepath), sr=None)

        # Compute tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

        # librosa returns tempo as numpy scalar
        bpm = float(tempo)

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

        logger.info(f"Detected BPM: {bpm:.1f} (confidence: {confidence:.2f})")
        return ComputedBPM(
            bpm=bpm, confidence=confidence, method="librosa", alternatives=alternatives
        )

    except Exception as e:
        logger.error(f"librosa BPM computation failed: {e}")
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"BPM computation failed: {e}")


def _adjust_bpm_to_edm_range(bpm: float, alternatives: list[float]) -> float:
    """Adjust BPM to preferred EDM range (120-150).

    If BPM is outside this range and there's an alternative within it,
    prefer the alternative.

    Parameters
    ----------
    bpm : float
        Primary BPM value.
    alternatives : list[float]
        Alternative BPM candidates.

    Returns
    -------
    float
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
            logger.debug(f"Adjusting BPM from {bpm:.1f} to {alt_bpm:.1f} (EDM range preference)")
            return alt_bpm

    # No alternative in range, return original
    return bpm


def compute_bpm(
    filepath: Path, prefer_madmom: bool = True, madmom_fps: int = 100, librosa_hop_length: int = 512
) -> ComputedBPM:
    """Compute BPM using available methods.

    Tries madmom first if preferred and available, falls back to librosa.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    prefer_madmom : bool, optional
        Try madmom first if True (default: True).
    madmom_fps : int, optional
        Frames per second for madmom (default: 100).
    librosa_hop_length : int, optional
        Hop length for librosa (default: 512).

    Returns
    -------
    ComputedBPM
        Computed BPM result.

    Raises
    ------
    AnalysisError
        If all computation methods fail.
    """
    if prefer_madmom:
        try:
            return compute_bpm_madmom(filepath, fps=madmom_fps)
        except Exception as e:
            logger.warning(f"madmom failed, falling back to librosa: {e}")
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length)
    else:
        try:
            return compute_bpm_librosa(filepath, hop_length=librosa_hop_length)
        except Exception as e:
            logger.warning(f"librosa failed, trying madmom: {e}")
            return compute_bpm_madmom(filepath, fps=madmom_fps)
