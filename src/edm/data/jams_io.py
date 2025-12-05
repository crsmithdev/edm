"""JAMS (JSON Annotated Music Specification) I/O utilities.

This module provides converters between our internal annotation schema
and the JAMS standard format used by the MIR research community.

References:
    - JAMS: https://github.com/marl/jams
    - Documentation: https://jams.readthedocs.io/
"""

from pathlib import Path
from typing import Optional

import jams  # type: ignore[import-not-found, import-untyped]

from edm.data.metadata import AnnotationTier
from edm.data.rekordbox import RekordboxTrack, extract_structure_boundaries
from edm.data.schema import Annotation


def rekordbox_to_jams(
    track: RekordboxTrack,
    tier: AnnotationTier = AnnotationTier.AUTO_CLEANED,
    confidence: float = 0.8,
    annotator_name: str = "rekordbox",
    cue_types: Optional[list[str]] = None,
    deduplicate: bool = True,
) -> jams.JAMS:
    """Convert Rekordbox track to JAMS format.

    Args:
        track: Rekordbox track data
        tier: Annotation tier (1=verified, 2=auto-cleaned, 3=auto-generated)
        confidence: Overall confidence score [0-1]
        annotator_name: Name of annotator/tool (default: "rekordbox")
        cue_types: Which cue types to include: ['hot'], ['memory'], ['all'].
                  Default ['hot'] to avoid duplicates from Rekordbox.
        deduplicate: Remove duplicate timestamps (default: True)

    Returns:
        JAMS object

    Raises:
        ValueError: If track data is invalid or incomplete
    """
    if cue_types is None:
        cue_types = ["hot"]  # Default to hot cues only

    # Validate required fields
    if track.bpm <= 0:
        raise ValueError(f"Invalid BPM: {track.bpm}")
    if track.duration <= 0:
        raise ValueError(f"Invalid duration: {track.duration}")
    if not track.cue_points:
        raise ValueError("Track has no cue points")

    # Filter cue points by type
    filtered_cues = _filter_cue_points(track.cue_points, cue_types)
    if not filtered_cues:
        raise ValueError(f"No cue points found matching types {cue_types}")

    # Create filtered track for boundary extraction
    filtered_track = RekordboxTrack(
        location=track.location,
        artist=track.artist,
        name=track.name,
        bpm=track.bpm,
        duration=track.duration,
        sample_rate=track.sample_rate,
        key=track.key,
        cue_points=filtered_cues,
        beat_grid=track.beat_grid,
    )

    # Extract structure boundaries
    boundaries = extract_structure_boundaries(filtered_track)
    if not boundaries:
        raise ValueError("No structure boundaries extracted from cue points")

    # Deduplicate if requested
    if deduplicate:
        boundaries = _deduplicate_boundaries(boundaries)

    # Create JAMS object
    jam = jams.JAMS()

    # Set file metadata
    jam.file_metadata.duration = track.duration
    jam.file_metadata.title = track.name
    jam.file_metadata.artist = track.artist

    # Add identifiers
    jam.file_metadata.identifiers = {"file_path": str(track.location)}

    # Create segment annotation
    ann = jams.Annotation(namespace="segment_open")

    # Set annotation metadata
    ann.annotation_metadata.curator = jams.Curator(name=annotator_name)
    ann.annotation_metadata.version = "1.0"
    ann.annotation_metadata.corpus = "rekordbox"
    ann.annotation_metadata.annotation_tools = "rekordbox"

    # Map tier to annotator type
    annotator_type_map = {
        AnnotationTier.VERIFIED: "human_verified",
        AnnotationTier.AUTO_CLEANED: "automatic_cleaned",
        AnnotationTier.AUTO_GENERATED: "automatic",
    }
    ann.annotation_metadata.annotator = {
        "output_type": annotator_type_map.get(tier, "automatic"),
        "tier": int(tier),
    }

    # Store import configuration in annotation rules
    ann.annotation_metadata.annotation_rules = (
        f"cue_types={cue_types}, deduplicate={deduplicate}, label_inference=cue_name"
    )

    # Add observations
    for time, label in boundaries:
        # JAMS segment_open uses time, duration=0, value=label, confidence
        ann.append(
            time=float(time),
            duration=0.0,  # Boundary marker, not a region
            value=label,
            confidence=float(confidence),
        )

    # Store additional track metadata in sandbox
    ann.sandbox.bpm = float(track.bpm)
    ann.sandbox.key = track.key
    ann.sandbox.sample_rate = track.sample_rate
    ann.sandbox.downbeat = float(track.beat_grid[0]) if track.beat_grid else 0.0
    ann.sandbox.time_signature = [4, 4]  # EDM default

    jam.annotations.append(ann)

    return jam


def annotation_to_jams(annotation: Annotation) -> jams.JAMS:
    """Convert our internal Annotation schema to JAMS format.

    Args:
        annotation: Internal annotation object

    Returns:
        JAMS object
    """
    jam = jams.JAMS()

    # Set file metadata
    jam.file_metadata.duration = float(annotation.audio.duration)
    jam.file_metadata.identifiers = {"file_path": str(annotation.audio.file)}

    # Create segment annotation
    ann = jams.Annotation(namespace="segment_open")

    # Set annotation metadata
    ann.annotation_metadata.curator = jams.Curator(
        name=annotation.metadata.verified_by or "unknown"
    )
    ann.annotation_metadata.version = "1.0"
    ann.annotation_metadata.corpus = annotation.metadata.source
    ann.annotation_metadata.annotation_tools = annotation.metadata.source

    # Map tier to annotator type
    annotator_type_map = {
        AnnotationTier.VERIFIED: "human_verified",
        AnnotationTier.AUTO_CLEANED: "automatic_cleaned",
        AnnotationTier.AUTO_GENERATED: "automatic",
    }
    ann.annotation_metadata.annotator = {
        "output_type": annotator_type_map.get(annotation.metadata.tier, "automatic"),
        "tier": int(annotation.metadata.tier),
    }

    # Add observations
    for section in annotation.structure:
        ann.append(
            time=float(section.time),
            duration=0.0,
            value=section.label,
            confidence=float(section.confidence),
        )

    # Store audio metadata in sandbox
    ann.sandbox.bpm = float(annotation.audio.bpm)
    ann.sandbox.key = annotation.audio.key
    ann.sandbox.downbeat = float(annotation.audio.downbeat)
    ann.sandbox.time_signature = list(annotation.audio.time_signature)
    ann.sandbox.bar_numbers = [int(s.bar) for s in annotation.structure]

    jam.annotations.append(ann)

    return jam


def jams_to_annotation(jam: jams.JAMS) -> Annotation:
    """Convert JAMS format to our internal Annotation schema.

    Args:
        jam: JAMS object

    Returns:
        Internal annotation object

    Raises:
        ValueError: If JAMS file has no segment annotations
    """
    from datetime import datetime, timezone

    from edm.data.metadata import AnnotationMetadata
    from edm.data.schema import AudioMetadata, StructureSection

    # Find segment annotation
    segment_anns = jam.search(namespace="segment.*")
    if not segment_anns:
        raise ValueError("No segment annotations found in JAMS file")

    # Use first segment annotation
    ann = segment_anns[0]

    # Extract tier from annotator metadata
    tier_map = {
        "human_verified": AnnotationTier.VERIFIED,
        "automatic_cleaned": AnnotationTier.AUTO_CLEANED,
        "automatic": AnnotationTier.AUTO_GENERATED,
    }
    # JAMS stores annotator as a dict-like JObject
    annotator_meta = ann.annotation_metadata.annotator
    if annotator_meta:
        output_type = (
            annotator_meta.get("output_type", "automatic")
            if hasattr(annotator_meta, "get")
            else getattr(annotator_meta, "output_type", "automatic")
        )
    else:
        output_type = "automatic"
    tier = tier_map.get(output_type, AnnotationTier.AUTO_GENERATED)

    # Calculate average confidence from observations
    if len(ann.data) > 0:
        confidences = [float(obs.confidence) for obs in ann.data]
        avg_confidence = sum(confidences) / len(confidences)
    else:
        avg_confidence = 1.0

    # Create metadata
    metadata = AnnotationMetadata(
        tier=tier,
        confidence=avg_confidence,
        source=ann.annotation_metadata.corpus or "unknown",
        created=datetime.now(timezone.utc),
        modified=datetime.now(timezone.utc),
        verified_by=ann.annotation_metadata.curator.name
        if ann.annotation_metadata.curator
        else None,
        notes=ann.annotation_metadata.annotation_rules,
        flags=[],
    )

    # Extract audio metadata (JAMS uses special Sandbox/JObject types)
    file_path = getattr(jam.file_metadata.identifiers, "file_path", "")
    bpm = float(getattr(ann.sandbox, "bpm", 128.0))
    downbeat = float(getattr(ann.sandbox, "downbeat", 0.0))
    time_sig = tuple(getattr(ann.sandbox, "time_signature", [4, 4]))
    key = getattr(ann.sandbox, "key", None)

    audio = AudioMetadata(
        file=Path(file_path),
        duration=float(jam.file_metadata.duration or 0.0),
        bpm=bpm,
        downbeat=downbeat,
        time_signature=time_sig,
        key=key,
    )

    # Extract structure sections
    bar_numbers = getattr(ann.sandbox, "bar_numbers", [])
    structure = []

    for i, obs in enumerate(ann.data):
        # Calculate bar number if not in sandbox
        if i < len(bar_numbers):
            bar = int(bar_numbers[i])
        else:
            # Fallback: calculate from time and BPM
            bar = _time_to_bar(obs.time, bpm, downbeat)

        structure.append(
            StructureSection(
                bar=bar,
                label=str(obs.value),
                time=float(obs.time),
                confidence=float(obs.confidence),
            )
        )

    return Annotation(metadata=metadata, audio=audio, structure=structure, energy=None)


def batch_convert_rekordbox_to_jams(
    xml_path: Path,
    output_dir: Path,
    tier: AnnotationTier = AnnotationTier.AUTO_CLEANED,
    confidence: float = 0.8,
    skip_existing: bool = True,
    cue_types: Optional[list[str]] = None,
    deduplicate: bool = True,
) -> tuple[int, int, list[str]]:
    """Batch convert Rekordbox XML to JAMS files.

    Args:
        xml_path: Path to rekordbox.xml file
        output_dir: Directory to save JAMS files (.jams extension)
        tier: Annotation tier for all tracks
        confidence: Confidence score for all tracks
        skip_existing: If True, skip tracks that already have JAMS files
        cue_types: Which cue types to include (default: ['hot'])
        deduplicate: Remove duplicate timestamps (default: True)

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
            output_path = output_dir / f"{safe_name}.jams"

            # Skip if exists
            if skip_existing and output_path.exists():
                skip_count += 1
                continue

            # Convert to JAMS
            jam = rekordbox_to_jams(
                track,
                tier=tier,
                confidence=confidence,
                cue_types=cue_types,
                deduplicate=deduplicate,
            )

            # Save to file (JAMS requires string path)
            jam.save(str(output_path))
            success_count += 1

        except Exception as e:
            errors.append(f"{track.name}: {e}")

    return success_count, skip_count, errors


def _filter_cue_points(cue_points: list, cue_types: list[str]) -> list:
    """Filter cue points by type.

    Args:
        cue_points: List of RekordboxCuePoint objects
        cue_types: List of cue types to include: 'hot', 'memory', 'all'

    Returns:
        Filtered list of cue points
    """
    if "all" in cue_types:
        return cue_points

    filtered = []
    for cue in cue_points:
        # Hot cues: Num >= 0 (typically 0-7)
        # Memory cues: Num = -1
        if "hot" in cue_types and cue.num >= 0:
            filtered.append(cue)
        elif "memory" in cue_types and cue.num == -1:
            filtered.append(cue)

    return filtered


def _deduplicate_boundaries(boundaries: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Remove duplicate boundaries by timestamp.

    Keeps the first occurrence of each unique timestamp.

    Args:
        boundaries: List of (time, label) tuples

    Returns:
        Deduplicated list
    """
    seen_times = set()
    deduplicated = []

    for time, label in boundaries:
        # Use small epsilon for float comparison (0.001s = 1ms)
        time_key = round(time, 3)
        if time_key not in seen_times:
            seen_times.add(time_key)
            deduplicated.append((time, label))

    return deduplicated


def _time_to_bar(time: float, bpm: float, downbeat: float) -> int:
    """Convert timestamp to 1-indexed bar number.

    Args:
        time: Timestamp in seconds
        bpm: Beats per minute
        downbeat: First downbeat time

    Returns:
        1-indexed bar number
    """
    adjusted_time = time - downbeat
    if adjusted_time < 0:
        return 1

    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4  # 4/4 time
    bar_index = int(adjusted_time / bar_duration)
    return max(1, bar_index + 1)


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
