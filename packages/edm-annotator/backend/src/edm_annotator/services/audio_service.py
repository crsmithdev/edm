"""Audio loading and validation service."""

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from edm.io.audio import load_audio


class AudioService:
    """Handles audio file loading and validation."""

    def __init__(self, config: Any):
        """Initialize audio service with configuration.

        Args:
            config: Flask configuration object with AUDIO_DIR
        """
        self.audio_dir = config["AUDIO_DIR"]
        self.sample_rate = config["WAVEFORM_SAMPLE_RATE"]
        self.audio_extensions = config["AUDIO_EXTENSIONS"]

    def validate_audio_path(self, filename: str) -> Path:
        """Validate and resolve audio file path.

        Args:
            filename: Audio filename (e.g., "track.mp3")

        Returns:
            Absolute path to audio file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path contains path traversal attempts
        """
        # Security: prevent path traversal
        if ".." in filename or filename.startswith("/"):
            raise ValueError(f"Invalid filename: {filename}")

        audio_path = self.audio_dir / filename

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {filename}")

        return audio_path

    def load_audio(self, filename: str) -> tuple[np.ndarray, int]:
        """Load audio file using edm library.

        Args:
            filename: Audio filename

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        audio_path = self.validate_audio_path(filename)
        return load_audio(audio_path, sr=self.sample_rate)

    def get_duration(self, filename: str) -> float:
        """Get audio duration in seconds.

        Args:
            filename: Audio filename

        Returns:
            Duration in seconds
        """
        audio_path = self.validate_audio_path(filename)
        return librosa.get_duration(path=audio_path)

    def list_audio_files(self) -> list[Path]:
        """List all audio files in audio directory.

        Returns:
            List of audio file paths
        """
        audio_files = []
        for ext in self.audio_extensions:
            audio_files.extend(self.audio_dir.glob(ext))
        return sorted(audio_files, key=lambda p: p.name)
