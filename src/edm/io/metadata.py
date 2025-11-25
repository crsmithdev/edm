"""Audio file metadata reading."""

from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from mutagen import File as MutagenFile

logger = structlog.get_logger(__name__)


def read_metadata(filepath: Path) -> Dict[str, Any]:
    """Read metadata from an audio file.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing metadata fields like artist, title, album,
        duration, bitrate, sample_rate, bpm, etc.

    Raises
    ------
    AudioFileError
        If the file cannot be read.

    Examples
    --------
    >>> from pathlib import Path
    >>> metadata = read_metadata(Path("track.mp3"))
    >>> print(f"{metadata['artist']} - {metadata['title']}")
    Artist Name - Track Title
    >>> print(f"BPM: {metadata.get('bpm', 'Unknown')}")
    BPM: 128.0
    """
    logger.info(f"Reading metadata from {filepath}")

    if not filepath.exists():
        from edm.exceptions import AudioFileError

        raise AudioFileError(f"File not found: {filepath}")

    try:
        audio = MutagenFile(filepath)
        if audio is None:
            raise ValueError(f"Unable to read file format: {filepath.suffix}")

        metadata = {
            "artist": _get_artist(audio),
            "title": _get_title(audio, filepath),
            "album": _get_album(audio),
            "duration": audio.info.length if hasattr(audio.info, "length") else None,
            "bitrate": getattr(audio.info, "bitrate", None),
            "sample_rate": getattr(audio.info, "sample_rate", None),
            "format": filepath.suffix[1:].upper(),
            "bpm": _get_bpm(audio, filepath),
        }

        logger.debug(f"Extracted metadata: {metadata}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to read metadata from {filepath}: {e}")
        from edm.exceptions import AudioFileError

        raise AudioFileError(f"Failed to read metadata: {e}")


def _get_artist(audio) -> Optional[str]:
    """Extract artist from audio file."""
    if hasattr(audio, "tags") and audio.tags:
        # ID3 tags (MP3)
        if "TPE1" in audio.tags:
            return str(audio.tags["TPE1"])
        # Vorbis comments (FLAC, OGG)
        if "artist" in audio.tags:
            val = audio.tags["artist"]
            return str(val[0]) if isinstance(val, list) else str(val)
        # MP4 tags
        if "©ART" in audio.tags:
            val = audio.tags["©ART"]
            return str(val[0]) if isinstance(val, list) else str(val)
    return None


def _get_title(audio, filepath: Path) -> str:
    """Extract title from audio file, falling back to filename."""
    if hasattr(audio, "tags") and audio.tags:
        # ID3 tags (MP3)
        if "TIT2" in audio.tags:
            return str(audio.tags["TIT2"])
        # Vorbis comments (FLAC, OGG)
        if "title" in audio.tags:
            val = audio.tags["title"]
            return str(val[0]) if isinstance(val, list) else str(val)
        # MP4 tags
        if "©nam" in audio.tags:
            val = audio.tags["©nam"]
            return str(val[0]) if isinstance(val, list) else str(val)
    return filepath.stem


def _get_album(audio) -> Optional[str]:
    """Extract album from audio file."""
    if hasattr(audio, "tags") and audio.tags:
        # ID3 tags (MP3)
        if "TALB" in audio.tags:
            return str(audio.tags["TALB"])
        # Vorbis comments (FLAC, OGG)
        if "album" in audio.tags:
            val = audio.tags["album"]
            return str(val[0]) if isinstance(val, list) else str(val)
        # MP4 tags
        if "©alb" in audio.tags:
            val = audio.tags["©alb"]
            return str(val[0]) if isinstance(val, list) else str(val)
    return None


def _get_bpm(audio, filepath: Path) -> Optional[float]:
    """Extract BPM from audio file metadata.

    Supports ID3v2 (MP3), MP4 (M4A, AAC), and FLAC formats.

    Parameters
    ----------
    audio : mutagen.FileType
        Mutagen audio file object.
    filepath : Path
        Path to the audio file (for logging).

    Returns
    -------
    Optional[float]
        BPM value if found and valid, None otherwise.
    """
    bpm = None

    try:
        if hasattr(audio, "tags") and audio.tags:
            # ID3 tags (MP3)
            if "TBPM" in audio.tags:
                bpm_val = str(audio.tags["TBPM"])
                bpm = float(bpm_val)
                logger.debug(f"Found BPM in ID3 tag: {bpm}")

            # Vorbis comments (FLAC, OGG) - try multiple tag names
            elif "bpm" in audio.tags:
                val = audio.tags["bpm"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug(f"Found BPM in vorbis comment: {bpm}")

            elif "tempo" in audio.tags:
                val = audio.tags["tempo"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug(f"Found BPM in tempo tag: {bpm}")

            # MP4 tags (M4A, AAC)
            elif "tmpo" in audio.tags:
                val = audio.tags["tmpo"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug(f"Found BPM in MP4 tag: {bpm}")

        # Validate BPM range
        if bpm is not None:
            if bpm <= 0 or bpm > 300:
                logger.warning(f"Invalid BPM value {bpm} in {filepath}, ignoring")
                return None

            return bpm

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse BPM from {filepath}: {e}")

    return None
