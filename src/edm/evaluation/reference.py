"""Reference data loading for accuracy evaluation."""

import csv
import json
from pathlib import Path
from typing import Any, Dict

import structlog

from edm.evaluation.common import discover_audio_files

logger = structlog.get_logger(__name__)


def load_reference_auto(
    reference_arg: str, analysis_type: str, source_path: Path, value_field: str = "bpm"
) -> Dict[Path, Any]:
    """Auto-detect and load reference based on argument and analysis type.

    Args:
        reference_arg: Reference source ('spotify', 'metadata', or file path)
        analysis_type: Type of analysis (bpm, drops, key)
        source_path: Source directory for audio files
        value_field: Field name for value extraction

    Returns:
        Dictionary mapping file paths to reference values

    Raises:
        ValueError: If reference type is not supported for analysis type
    """
    logger.info(
        "loading reference",
        reference_arg=reference_arg,
        analysis_type=analysis_type,
        value_field=value_field,
    )

    # Handle special reference types
    if reference_arg.lower() == "spotify":
        if analysis_type not in ["bpm"]:
            raise ValueError(
                f"Spotify reference not supported for '{analysis_type}' analysis. "
                f"Only 'bpm' is supported."
            )
        return load_spotify_reference(source_path)

    if reference_arg.lower() == "metadata":
        if analysis_type not in ["bpm", "key"]:
            raise ValueError(
                f"Metadata reference not supported for '{analysis_type}' analysis. "
                f"Only 'bpm' and 'key' are supported."
            )
        return load_metadata_reference(source_path, value_field)

    # Handle file-based references
    ref_path = Path(reference_arg)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    if ref_path.suffix == ".csv":
        return load_reference_csv(ref_path, value_field)
    elif ref_path.suffix == ".json":
        return load_reference_json(ref_path, value_field)
    else:
        raise ValueError(
            f"Unknown reference format: {reference_arg}. "
            f"Expected .csv, .json, 'spotify', or 'metadata'"
        )


def load_reference_csv(path: Path, value_field: str = "bpm") -> Dict[Path, float]:
    """Load reference data from CSV file.

    Expected format:
        path,bpm
        /music/track1.mp3,128.0
        /music/track2.flac,140.0

    Args:
        path: Path to CSV file
        value_field: Name of the value column

    Returns:
        Dictionary mapping file paths to reference values
    """
    reference = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        if "path" not in reader.fieldnames:
            raise ValueError("CSV must have 'path' column")

        if value_field not in reader.fieldnames:
            raise ValueError(f"CSV must have '{value_field}' column")

        for row in reader:
            file_path = Path(row["path"]).resolve()
            try:
                reference[file_path] = float(row[value_field])
            except ValueError:
                logger.warning(
                    "invalid value", file=str(file_path), value=row[value_field], field=value_field
                )

    logger.info(
        "loaded reference csv", count=len(reference), path=str(path), value_field=value_field
    )

    return reference


def load_reference_json(path: Path, value_field: str = "bpm") -> Dict[Path, Any]:
    """Load reference data from JSON file.

    Expected format:
        [
          {"path": "/music/track1.mp3", "bpm": 128.0},
          {"path": "/music/track2.flac", "bpm": 140.0}
        ]

    Args:
        path: Path to JSON file
        value_field: Name of the value field

    Returns:
        Dictionary mapping file paths to reference values
    """
    reference = {}

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects")

    for item in data:
        if "path" not in item:
            logger.warning("missing path field", item=item)
            continue

        if value_field not in item:
            logger.warning("missing value field", path=item["path"], field=value_field)
            continue

        file_path = Path(item["path"]).resolve()
        reference[file_path] = item[value_field]

    logger.info(
        "loaded reference json", count=len(reference), path=str(path), value_field=value_field
    )

    return reference


def load_spotify_reference(source_path: Path) -> Dict[Path, float]:
    """Load BPM data from Spotify API for discovered files.

    Args:
        source_path: Source directory for audio files

    Returns:
        Dictionary mapping file paths to BPM values
    """
    from edm.external.spotify import SpotifyClient
    from edm.io.metadata import read_metadata

    client = SpotifyClient()
    reference = {}

    files = discover_audio_files(source_path)

    logger.info("fetching spotify reference", file_count=len(files))

    for file_path in files:
        try:
            metadata = read_metadata(file_path)
            artist = metadata.get("artist")
            title = metadata.get("title")

            if not artist or not title:
                logger.debug("missing metadata", file=str(file_path), artist=artist, title=title)
                continue

            track_info = client.search_track(artist, title)
            if track_info and track_info.get("bpm"):
                reference[file_path] = float(track_info["bpm"])
                logger.debug("spotify lookup success", file=file_path.name, bpm=track_info["bpm"])

        except Exception as e:
            logger.warning("spotify lookup failed", file=str(file_path), error=str(e))

    logger.info("loaded spotify reference", count=len(reference))

    return reference


def load_metadata_reference(source_path: Path, value_field: str = "bpm") -> Dict[Path, Any]:
    """Load reference data from file metadata (ID3/Vorbis/MP4 tags).

    Args:
        source_path: Source directory for audio files
        value_field: Metadata field to extract (bpm, key, etc.)

    Returns:
        Dictionary mapping file paths to reference values
    """
    from edm.io.metadata import read_metadata

    reference = {}
    files = discover_audio_files(source_path)

    logger.info("reading metadata reference", file_count=len(files), value_field=value_field)

    for file_path in files:
        try:
            metadata = read_metadata(file_path)

            if value_field in metadata and metadata[value_field]:
                # Convert to appropriate type based on field
                if value_field == "bpm":
                    reference[file_path] = float(metadata[value_field])
                elif value_field == "key":
                    reference[file_path] = str(metadata[value_field])
                else:
                    reference[file_path] = metadata[value_field]

                logger.debug(
                    "metadata read success", file=file_path.name, value=reference[file_path]
                )

        except Exception as e:
            logger.warning(
                "metadata read failed", file=str(file_path), field=value_field, error=str(e)
            )

    logger.info("loaded metadata reference", count=len(reference), value_field=value_field)

    return reference
