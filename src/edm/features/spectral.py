"""Spectral feature extraction."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectralFeatures:
    """Spectral features extracted from audio.

    Attributes
    ----------
    centroid : np.ndarray
        Spectral centroid over time.
    rolloff : np.ndarray
        Spectral rolloff over time.
    flux : np.ndarray
        Spectral flux over time.
    """
    centroid: np.ndarray
    rolloff: np.ndarray
    flux: np.ndarray


def extract_spectral_features(
    audio_data: np.ndarray,
    sample_rate: int
) -> SpectralFeatures:
    """Extract spectral features from audio.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio samples.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    SpectralFeatures
        Extracted spectral features.

    Examples
    --------
    >>> features = extract_spectral_features(audio, sr)
    >>> print(f"Mean centroid: {features.centroid.mean():.1f} Hz")
    """
    logger.debug(f"Extracting spectral features from {len(audio_data)} samples")

    # TODO: Implement actual feature extraction with librosa
    # Placeholder implementation
    num_frames = 100
    return SpectralFeatures(
        centroid=np.zeros(num_frames),
        rolloff=np.zeros(num_frames),
        flux=np.zeros(num_frames)
    )
