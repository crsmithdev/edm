"""Beat detection using beat_this and librosa.

This module provides beat and downbeat detection, returning a BeatGrid
for index-based beat position representation.
"""

from pathlib import Path

import librosa
import numpy as np
import structlog

from edm.analysis.beat_grid import BeatGrid
from edm.io.audio import AudioData, load_audio

logger = structlog.get_logger(__name__)


def detect_beats_beat_this(filepath: Path, device: str | None = None) -> BeatGrid:
    """Detect beats using beat_this neural network.

    Args:
        filepath: Path to the audio file.
        device: Device for inference ('cuda', 'cpu', or None for auto-detect).

    Returns:
        BeatGrid with parameters derived from beat_this inference.

    Raises:
        AnalysisError: If beat detection fails.
    """
    logger.debug("detecting beats with beat_this", filepath=str(filepath))

    try:
        import torch
        from beat_this.inference import File2Beats

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.debug("beat_this using device", device=device)

        # Initialize beat tracker
        file2beats = File2Beats(checkpoint_path="final0", device=device)

        # Get beat and downbeat positions
        beats, downbeats = file2beats(str(filepath))

        beats = np.array(beats)
        downbeats = np.array(downbeats) if downbeats is not None else None

        if len(beats) < 2:
            logger.warning("insufficient beats detected", beat_count=len(beats))
            from edm.exceptions import AnalysisError

            raise AnalysisError(f"Insufficient beats detected: {len(beats)}")

        # Calculate confidence based on interval consistency
        intervals = np.diff(beats)
        median_interval = float(np.median(intervals))
        std = float(np.std(intervals))
        confidence = max(0.0, min(1.0, 1.0 - (std / median_interval)))

        # Create BeatGrid from timestamps
        grid = BeatGrid.from_timestamps(
            beats=beats,
            downbeats=downbeats,
            confidence=confidence,
            method="beat-this",
        )

        logger.debug(
            "detected beat grid",
            first_beat=round(grid.first_beat_time, 3),
            bpm=round(grid.bpm, 1),
            confidence=round(confidence, 2),
        )
        return grid

    except ImportError as e:
        logger.error("beat_this not installed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError("beat_this library not installed")
    except Exception as e:
        logger.error("beat_this detection failed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"Beat detection failed: {e}")


def detect_beats_librosa(
    filepath: Path,
    hop_length: int = 512,
    audio: AudioData | None = None,
) -> BeatGrid:
    """Detect beats using librosa.

    Librosa does not provide native downbeat detection, so the first downbeat
    is estimated using beat energy analysis (highest energy beat in 4-beat groups).

    Args:
        filepath: Path to the audio file.
        hop_length: Hop length for analysis (default: 512).
        audio: Pre-loaded audio data as (y, sr) tuple.

    Returns:
        BeatGrid with parameters derived from librosa.

    Raises:
        AnalysisError: If beat detection fails.
    """
    logger.debug("detecting beats with librosa", filepath=str(filepath))

    try:
        # Use provided audio or load from disk
        if audio is not None:
            y, sr = audio
        else:
            y, sr = load_audio(filepath)

        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        if len(beat_times) < 2:
            logger.warning("insufficient beats detected", beat_count=len(beat_times))
            from edm.exceptions import AnalysisError

            raise AnalysisError(f"Insufficient beats detected: {len(beat_times)}")

        # Estimate downbeats using energy analysis
        # Find which beat position (0-3) has highest average energy
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        beat_energies = onset_env[beat_frames] if len(beat_frames) > 0 else np.array([])

        downbeat_times = _estimate_downbeats_from_energy(beat_times, beat_energies)

        # Calculate confidence
        intervals = np.diff(beat_times)
        std = float(np.std(intervals))
        mean_interval = float(np.mean(intervals))
        confidence = max(0.0, min(1.0, 1.0 - (std / mean_interval)))

        # Create BeatGrid
        grid = BeatGrid.from_timestamps(
            beats=beat_times,
            downbeats=downbeat_times if len(downbeat_times) > 0 else None,
            confidence=confidence,
            method="librosa",
        )

        logger.debug(
            "detected beat grid",
            first_beat=round(grid.first_beat_time, 3),
            bpm=round(grid.bpm, 1),
            confidence=round(confidence, 2),
        )
        return grid

    except Exception as e:
        logger.error("librosa detection failed", error=str(e))
        from edm.exceptions import AnalysisError

        raise AnalysisError(f"Beat detection failed: {e}")


def _estimate_downbeats_from_energy(
    beat_times: np.ndarray,
    beat_energies: np.ndarray,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """Estimate downbeat positions from beat energy analysis.

    Assumes the downbeat (beat 1 of each bar) typically has higher energy.
    Groups beats into bars and finds which beat position has highest
    cumulative energy across the track.

    Args:
        beat_times: Array of beat timestamps.
        beat_energies: Array of onset energy at each beat.
        beats_per_bar: Number of beats per bar (default 4 for 4/4 time).

    Returns:
        Array of estimated downbeat timestamps.
    """
    if len(beat_times) < beats_per_bar or len(beat_energies) < beats_per_bar:
        # Not enough beats to estimate, return first beat as downbeat
        return beat_times[::beats_per_bar] if len(beat_times) > 0 else np.array([])

    # Calculate cumulative energy for each beat position (0, 1, 2, 3)
    position_energies = np.zeros(beats_per_bar)
    for i, energy in enumerate(beat_energies):
        position = i % beats_per_bar
        position_energies[position] += energy

    # Find position with highest energy (likely the downbeat)
    downbeat_position = int(np.argmax(position_energies))

    # Extract beats at downbeat positions
    downbeat_indices = np.arange(downbeat_position, len(beat_times), beats_per_bar)
    return np.asarray(beat_times[downbeat_indices])


def detect_beats(
    filepath: Path,
    device: str | None = None,
    hop_length: int = 512,
    audio: AudioData | None = None,
) -> BeatGrid:
    """Detect beats using available methods.

    Tries beat_this first if available, falls back to librosa.

    Args:
        filepath: Path to the audio file.
        device: Device for beat_this inference ('cuda', 'cpu', or None for auto).
        hop_length: Hop length for librosa (default: 512).
        audio: Pre-loaded audio data (only used by librosa fallback).

    Returns:
        BeatGrid with detected beat parameters.

    Raises:
        AnalysisError: If all detection methods fail.
    """
    try:
        return detect_beats_beat_this(filepath, device=device)
    except Exception as e:
        logger.warning("beat_this failed, falling back to librosa", error=str(e))
        return detect_beats_librosa(filepath, hop_length=hop_length, audio=audio)
