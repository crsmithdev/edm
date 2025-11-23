"""Audio file loading."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(filepath: Path, *, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """Load an audio file.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    sample_rate : int, optional
        Target sample rate (default: 44100 Hz).

    Returns
    -------
    audio_data : np.ndarray
        Audio samples as numpy array.
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    AudioFileError
        If the file cannot be loaded or format is unsupported.

    Examples
    --------
    >>> from pathlib import Path
    >>> audio, sr = load_audio(Path("track.mp3"))
    >>> print(f"Loaded {len(audio)} samples at {sr} Hz")
    Loaded 7938000 samples at 44100 Hz
    """
    logger.info(f"Loading audio from {filepath}")
    logger.debug(f"Target sample rate: {sample_rate} Hz")

    if not filepath.exists():
        from edm.exceptions import AudioFileError
        raise AudioFileError(f"File not found: {filepath}")

    # TODO: Implement actual audio loading with librosa
    # Placeholder implementation
    duration = 180.0  # 3 minutes
    num_samples = int(duration * sample_rate)
    audio_data = np.zeros(num_samples, dtype=np.float32)

    return audio_data, sample_rate
