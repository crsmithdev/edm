"""File I/O module for audio files and metadata."""

from edm.io.files import SUPPORTED_AUDIO_FORMATS, discover_audio_files
from edm.io.metadata import read_metadata

__all__ = ["SUPPORTED_AUDIO_FORMATS", "discover_audio_files", "read_metadata"]
