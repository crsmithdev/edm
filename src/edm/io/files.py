"""Audio file discovery utilities."""

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Canonical set of supported audio file extensions
SUPPORTED_AUDIO_FORMATS = frozenset({".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"})


def discover_audio_files(
    paths: Path | list[Path],
    *,
    recursive: bool = True,
) -> list[Path]:
    """Discover audio files from given paths.

    Handles both individual files and directories. For directories,
    searches for all supported audio formats.

    Args:
        paths: Single path or list of file/directory paths to search.
        recursive: Search directories recursively. Defaults to True.

    Returns:
        Sorted list of audio file paths.

    Raises:
        FileNotFoundError: If a path does not exist.

    Examples:
        >>> # Single directory, recursive
        >>> files = discover_audio_files(Path("/music"))

        >>> # Multiple paths, non-recursive
        >>> files = discover_audio_files(
        ...     [Path("/music/track.mp3"), Path("/albums")],
        ...     recursive=False,
        ... )
    """
    # Normalize to list
    if isinstance(paths, Path):
        paths = [paths]

    audio_files: list[Path] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                audio_files.append(path)
            else:
                logger.debug(
                    "skipping non-audio file",
                    path=str(path),
                    suffix=path.suffix,
                )
        elif path.is_dir():
            glob_method = path.rglob if recursive else path.glob
            pattern = "*"

            for ext in SUPPORTED_AUDIO_FORMATS:
                audio_files.extend(glob_method(f"{pattern}{ext}"))

    # Sort for reproducibility
    audio_files.sort()

    logger.info(
        "discovered audio files",
        count=len(audio_files),
        paths=[str(p) for p in paths],
        recursive=recursive,
    )

    return audio_files
