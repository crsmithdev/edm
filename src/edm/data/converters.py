"""Converters for transforming external formats to Annotation schema."""

from datetime import datetime, timezone
from pathlib import Path

from edm.data.metadata import AnnotationMetadata, AnnotationTier
from edm.data.rekordbox import RekordboxTrack, extract_structure_boundaries
from edm.data.schema import Annotation, AudioMetadata, StructureSection


def rekordbox_to_annotation(
    track: RekordboxTrack,
    tier: AnnotationTier = AnnotationTier.AUTO_CLEANED,
    confidence: float = 0.8,
    verified_by: str | None = None,
    notes: str | None = None,
) -> Annotation:
    """Convert Rekordbox track to Annotation schema.

    Args:
        track: Rekordbox track data
        tier: Annotation tier (default: AUTO_CLEANED)
        confidence: Overall confidence score [0-1]
        verified_by: Optional username who verified this annotation
        notes: Optional notes about this annotation

    Returns:
        Annotation object

    Raises:
        ValueError: If track data is invalid or incomplete
    """
    # Validate required fields
    if track.bpm <= 0:
        raise ValueError(f"Invalid BPM: {track.bpm}")
    if track.duration <= 0:
        raise ValueError(f"Invalid duration: {track.duration}")
    if not track.cue_points:
        raise ValueError("Track has no cue points")

    # Extract structure boundaries from cue points
    boundaries = extract_structure_boundaries(track)
    if not boundaries:
        raise ValueError("No structure boundaries extracted from cue points")

    # Convert to StructureSection objects
    # Calculate bar numbers from beat grid and timestamps
    structure_sections = []
    for time, label in boundaries:
        bar = _time_to_bar(time, track.bpm, track.beat_grid)
        structure_sections.append(
            StructureSection(
                bar=bar,
                label=label,
                time=time,
                confidence=confidence,
            )
        )

    # Sort by time
    structure_sections = sorted(structure_sections, key=lambda s: s.time)

    # Determine downbeat (first beat)
    downbeat = track.beat_grid[0] if track.beat_grid else 0.0

    # Create metadata
    now = datetime.now(timezone.utc)
    metadata = AnnotationMetadata(
        tier=tier,
        confidence=confidence,
        source="rekordbox",
        created=now,
        modified=now,
        verified_by=verified_by,
        notes=notes,
        flags=[],
    )

    # Create audio metadata
    audio = AudioMetadata(
        file=track.location,
        duration=track.duration,
        bpm=track.bpm,
        downbeat=downbeat,
        time_signature=(4, 4),  # Default for EDM
        key=track.key,
    )

    return Annotation(
        metadata=metadata,
        audio=audio,
        structure=structure_sections,
        energy=None,  # Energy computed separately
    )


def _time_to_bar(time: float, bpm: float, beat_grid: list[float]) -> int:
    """Convert timestamp to 1-indexed bar number.

    Args:
        time: Timestamp in seconds
        bpm: Beats per minute
        beat_grid: List of beat times

    Returns:
        1-indexed bar number

    Notes:
        Uses beat grid if available, otherwise computes from BPM.
        Assumes 4/4 time signature (4 beats per bar).
    """
    if not beat_grid:
        # Compute from BPM
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * 4  # 4 beats per bar
        bar_index = int(time / bar_duration)
        return max(1, bar_index + 1)  # 1-indexed, minimum bar 1

    # Find nearest beat in grid
    if time <= beat_grid[0]:
        return 1

    # Find beat index
    beat_index = 0
    for i, beat_time in enumerate(beat_grid):
        if beat_time <= time:
            beat_index = i
        else:
            break

    # Convert beat index to bar (4 beats per bar, 1-indexed)
    bar = (beat_index // 4) + 1
    return max(1, bar)


def batch_convert_rekordbox_xml(
    xml_path: Path,
    output_dir: Path,
    tier: AnnotationTier = AnnotationTier.AUTO_CLEANED,
    confidence: float = 0.8,
    skip_existing: bool = True,
) -> tuple[int, int, list[str]]:
    """Batch convert Rekordbox XML to Annotation YAML files.

    Args:
        xml_path: Path to rekordbox.xml file
        output_dir: Directory to save annotation YAML files
        tier: Annotation tier for all tracks
        confidence: Confidence score for all tracks
        skip_existing: If True, skip tracks that already have annotations

    Returns:
        Tuple of (success_count, skip_count, error_list)
    """
    from edm.data.rekordbox import parse_rekordbox_xml

    # Parse XML
    tracks = parse_rekordbox_xml(xml_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    errors = []

    for track in tracks:
        try:
            # Generate output filename from track name
            safe_name = _safe_filename(f"{track.artist} - {track.name}")
            output_path = output_dir / f"{safe_name}.yaml"

            # Skip if exists
            if skip_existing and output_path.exists():
                skip_count += 1
                continue

            # Convert to annotation
            annotation = rekordbox_to_annotation(
                track,
                tier=tier,
                confidence=confidence,
            )

            # Save to YAML
            annotation.to_yaml(output_path)
            success_count += 1

        except Exception as e:
            errors.append(f"{track.name}: {e}")

    return success_count, skip_count, errors


def _safe_filename(name: str, max_length: int = 200) -> str:
    """Convert string to safe filename.

    Args:
        name: Original filename
        max_length: Maximum filename length

    Returns:
        Safe filename string
    """
    # Replace unsafe characters
    safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace("<", "_").replace(">", "_").replace("|", "_")
    safe = safe.replace("?", "_").replace("*", "_").replace('"', "_")

    # Trim to max length
    if len(safe) > max_length:
        safe = safe[:max_length]

    return safe.strip()
