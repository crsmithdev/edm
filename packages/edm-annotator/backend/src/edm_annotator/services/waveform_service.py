"""Waveform generation and processing service."""

from typing import Any

import librosa
import numpy as np
from scipy import signal

from .audio_service import AudioService


class WaveformService:
    """Handles 3-band waveform generation and DSP processing."""

    def __init__(self, config: Any, audio_service: AudioService):
        """Initialize waveform service.

        Args:
            config: Flask configuration object with DSP parameters
            audio_service: AudioService instance for loading audio
        """
        self.audio_service = audio_service
        self.sample_rate = config["WAVEFORM_SAMPLE_RATE"]
        self.hop_length = config["WAVEFORM_HOP_LENGTH"]
        self.frame_length = config["WAVEFORM_FRAME_LENGTH"]
        self.bass_low = config["BASS_LOW"]
        self.bass_high = config["BASS_HIGH"]
        self.mids_low = config["MIDS_LOW"]
        self.mids_high = config["MIDS_HIGH"]
        self.highs_low = config["HIGHS_LOW"]

    def generate_waveform(self, filename: str) -> dict[str, Any]:
        """Generate 3-band RMS waveform for visualization.

        Args:
            filename: Audio filename

        Returns:
            Dictionary containing:
                - waveform_bass: List of RMS values for bass band (20-250 Hz)
                - waveform_mids: List of RMS values for mids band (250-4000 Hz)
                - waveform_highs: List of RMS values for highs band (4000+ Hz)
                - waveform_times: List of frame times in seconds
                - duration: Total duration in seconds
                - sample_rate: Sample rate used
        """
        # Load audio
        y, sr = self.audio_service.load_audio(filename)
        duration = len(y) / sr

        # Apply bandpass filters to isolate frequency bands
        y_bass = self._apply_bandpass_filter(y, sr, self.bass_low, self.bass_high)
        y_mids = self._apply_bandpass_filter(y, sr, self.mids_low, self.mids_high)
        y_highs = self._apply_highpass_filter(y, sr, self.highs_low)

        # Calculate RMS energy for each band
        rms_bass = self._compute_rms(y_bass, self.frame_length, self.hop_length)
        rms_mids = self._compute_rms(y_mids, self.frame_length, self.hop_length)
        rms_highs = self._compute_rms(y_highs, self.frame_length, self.hop_length)

        # Generate time axis for RMS frames
        times = librosa.frames_to_time(np.arange(len(rms_bass)), sr=sr, hop_length=self.hop_length)

        return {
            "waveform_bass": rms_bass.tolist(),
            "waveform_mids": rms_mids.tolist(),
            "waveform_highs": rms_highs.tolist(),
            "waveform_times": times.tolist(),
            "duration": duration,
            "sample_rate": sr,
        }

    def _apply_bandpass_filter(
        self, y: np.ndarray, sr: int, low_freq: float, high_freq: float
    ) -> np.ndarray:
        """Apply butterworth bandpass filter to audio signal.

        Args:
            y: Audio signal
            sr: Sample rate
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz

        Returns:
            Filtered audio signal
        """
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Design 4th-order Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype="band")

        # Apply zero-phase filter (forward and backward)
        return signal.filtfilt(b, a, y)

    def _apply_highpass_filter(self, y: np.ndarray, sr: int, cutoff_freq: float) -> np.ndarray:
        """Apply butterworth highpass filter to audio signal.

        Args:
            y: Audio signal
            sr: Sample rate
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Filtered audio signal
        """
        nyquist = sr / 2
        cutoff = cutoff_freq / nyquist

        # Design 4th-order Butterworth highpass filter
        b, a = signal.butter(4, cutoff, btype="high")

        # Apply zero-phase filter (forward and backward)
        return signal.filtfilt(b, a, y)

    def _compute_rms(self, y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Compute root mean square (RMS) energy for audio signal.

        Args:
            y: Audio signal
            frame_length: Frame length for RMS calculation
            hop_length: Hop length between frames

        Returns:
            1D array of RMS values
        """
        return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
