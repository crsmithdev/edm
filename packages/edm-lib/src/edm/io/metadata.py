"""Audio file metadata reading."""

from pathlib import Path
from typing import Any

import structlog
from mutagen import File as MutagenFile

logger = structlog.get_logger(__name__)


def read_metadata(filepath: Path) -> dict[str, Any]:
    """Read metadata from an audio file.

    Args:
        filepath: Path to the audio file.

    Returns:
        Dictionary containing metadata fields like artist, title, album,
        duration, bitrate, sample_rate, bpm, etc.

    Raises:
        AudioFileError: If the file cannot be read.

    Examples:
        >>> from pathlib import Path
        >>> metadata = read_metadata(Path("track.mp3"))
        >>> print(f"{metadata['artist']} - {metadata['title']}")
        Artist Name - Track Title
        >>> print(f"BPM: {metadata.get('bpm', 'Unknown')}")
        BPM: 128.0
    """
    logger.debug("reading metadata", filepath=str(filepath))

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

        logger.debug("extracted metadata", metadata=metadata)
        return metadata

    except Exception as e:
        logger.error("failed to read metadata", filepath=str(filepath), error=str(e))
        from edm.exceptions import AudioFileError

        raise AudioFileError(f"Failed to read metadata: {e}")


def _get_artist(audio) -> str | None:
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


def _get_album(audio) -> str | None:
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


def _get_bpm(audio, filepath: Path) -> float | None:
    """Extract BPM from audio file metadata.

    Supports ID3v2 (MP3), MP4 (M4A, AAC), and FLAC formats.

    Args:
        audio: Mutagen audio file object.
        filepath: Path to the audio file (for logging).

    Returns:
        BPM value if found and valid, None otherwise.
    """
    bpm = None

    try:
        if hasattr(audio, "tags") and audio.tags:
            # ID3 tags (MP3)
            if "TBPM" in audio.tags:
                bpm_val = str(audio.tags["TBPM"])
                bpm = float(bpm_val)
                logger.debug("found bpm in id3 tag", bpm=bpm)

            # Vorbis comments (FLAC, OGG) - try multiple tag names
            elif "bpm" in audio.tags:
                val = audio.tags["bpm"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug("found bpm in vorbis comment", bpm=bpm)

            elif "tempo" in audio.tags:
                val = audio.tags["tempo"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug("found bpm in tempo tag", bpm=bpm)

            # MP4 tags (M4A, AAC)
            elif "tmpo" in audio.tags:
                val = audio.tags["tmpo"]
                bpm_val = val[0] if isinstance(val, list) else val
                bpm = float(bpm_val)
                logger.debug("found bpm in mp4 tag", bpm=bpm)

        # Validate BPM range
        if bpm is not None:
            if bpm <= 0 or bpm > 300:
                logger.warning("invalid bpm value, ignoring", bpm=bpm, filepath=str(filepath))
                return None

            return bpm

    except (ValueError, TypeError) as e:
        logger.warning("failed to parse bpm", filepath=str(filepath), error=str(e))

    return None
