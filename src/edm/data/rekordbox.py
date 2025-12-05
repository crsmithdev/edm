"""Rekordbox XML parser for extracting cue points and beat grids."""

from pathlib import Path
from typing import Optional
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as DefusedET  # type: ignore[import-untyped]
from pydantic import BaseModel


class RekordboxCuePoint(BaseModel):
    """A single cue point from Rekordbox.

    Attributes:
        name: Cue point name/label
        time: Time in seconds
        type: Cue type (0=memory, 1=hot cue, etc.)
        num: Cue point number
    """

    name: str
    time: float
    type: int
    num: int


class RekordboxTrack(BaseModel):
    """Track data parsed from Rekordbox XML.

    Attributes:
        location: File path to audio file
        artist: Track artist
        name: Track name
        bpm: Beats per minute
        duration: Track duration in seconds
        sample_rate: Audio sample rate
        key: Musical key (Camelot notation if available)
        cue_points: List of cue points
        beat_grid: List of beat times in seconds
    """

    location: Path
    artist: str
    name: str
    bpm: float
    duration: float
    sample_rate: int
    key: Optional[str] = None
    cue_points: list[RekordboxCuePoint]
    beat_grid: list[float]


def parse_rekordbox_xml(xml_path: Path) -> list[RekordboxTrack]:
    """Parse Rekordbox XML export file.

    Args:
        xml_path: Path to rekordbox.xml file

    Returns:
        List of RekordboxTrack objects

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ValueError: If XML is invalid or malformed
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Rekordbox XML not found: {xml_path}")

    try:
        tree = DefusedET.parse(xml_path)
        root = tree.getroot()
    except DefusedET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    tracks = []

    # Find all TRACK elements in the COLLECTION
    collection = root.find(".//COLLECTION")
    if collection is None:
        raise ValueError("No COLLECTION element found in XML")

    for track_elem in collection.findall("TRACK"):
        try:
            track = _parse_track_element(track_elem)
            tracks.append(track)
        except (ValueError, KeyError):
            # Skip tracks with missing required fields
            continue

    return tracks


def _parse_track_element(track_elem: Element) -> RekordboxTrack:
    """Parse a single TRACK element.

    Args:
        track_elem: XML element for track

    Returns:
        RekordboxTrack object

    Raises:
        ValueError: If required fields are missing
        KeyError: If required attributes don't exist
    """
    # Extract basic track info from attributes
    location = track_elem.get("Location")
    if not location:
        raise ValueError("Track missing Location attribute")

    # Handle file:// URLs and convert to Path
    if location.startswith("file://localhost/"):
        location = location.replace("file://localhost/", "/")
    elif location.startswith("file://"):
        location = location.replace("file://", "")

    artist = track_elem.get("Artist", "Unknown Artist")
    name = track_elem.get("Name", "Unknown Track")
    bpm_str = track_elem.get("AverageBpm", track_elem.get("Bpm", "0"))
    duration_str = track_elem.get("TotalTime", "0")
    sample_rate_str = track_elem.get("SampleRate", "44100")
    key = track_elem.get("Tonality")  # Musical key

    # Parse numeric fields
    bpm = float(bpm_str)
    duration = float(duration_str)  # Already in seconds
    sample_rate = int(sample_rate_str)

    # Parse cue points
    cue_points = []
    position_marks = track_elem.findall(".//POSITION_MARK")
    for mark in position_marks:
        # Type 0 = memory cue, Type 1 = hot cue
        mark_type = int(mark.get("Type", "0"))
        start_str = mark.get("Start")
        if start_str is None:
            continue

        start_time = float(start_str)
        mark_name = mark.get("Name", f"Cue {mark.get('Num', '?')}")
        mark_num = int(mark.get("Num", "-1"))

        cue_points.append(
            RekordboxCuePoint(
                name=mark_name,
                time=start_time,
                type=mark_type,
                num=mark_num,
            )
        )

    # Parse beat grid (TEMPO elements)
    beat_grid = []
    tempo_elements = track_elem.findall(".//TEMPO")
    if tempo_elements:
        # Extract beat positions from TEMPO markers
        for tempo in tempo_elements:
            inizio = tempo.get("Inizio")  # Italian for "start"
            if inizio:
                beat_time = float(inizio)
                beat_grid.append(beat_time)

    # If no explicit beat grid, generate from BPM and duration
    if not beat_grid and bpm > 0 and duration > 0:
        beat_grid = _generate_beat_grid_from_bpm(bpm, duration)

    return RekordboxTrack(
        location=Path(location),
        artist=artist,
        name=name,
        bpm=bpm,
        duration=duration,
        sample_rate=sample_rate,
        key=key,
        cue_points=cue_points,
        beat_grid=sorted(beat_grid),  # Ensure chronological order
    )


def _generate_beat_grid_from_bpm(bpm: float, duration: float, downbeat: float = 0.0) -> list[float]:
    """Generate evenly-spaced beat grid from BPM.

    Args:
        bpm: Beats per minute
        duration: Track duration in seconds
        downbeat: First downbeat position in seconds

    Returns:
        List of beat times in seconds
    """
    beat_interval = 60.0 / bpm
    beat_grid = []
    current_time = downbeat

    while current_time <= duration:
        beat_grid.append(current_time)
        current_time += beat_interval

    return beat_grid


def extract_structure_boundaries(
    track: RekordboxTrack,
    cue_filter: Optional[list[str]] = None,
) -> list[tuple[float, str]]:
    """Extract structure boundaries from cue points.

    Args:
        track: Rekordbox track data
        cue_filter: Optional list of cue names to include (case-insensitive).
                   If None, all cues are included.

    Returns:
        List of (time, label) tuples for structure boundaries
    """
    boundaries = []

    for cue in track.cue_points:
        # Filter by cue name if specified
        if cue_filter:
            if not any(filter_name.lower() in cue.name.lower() for filter_name in cue_filter):
                continue

        # Try to infer section label from cue name
        label = _infer_section_label(cue.name)
        boundaries.append((cue.time, label))

    return sorted(boundaries, key=lambda x: x[0])


def _infer_section_label(cue_name: str) -> str:
    """Infer EDM section label from cue point name.

    Args:
        cue_name: Cue point name

    Returns:
        Section label (intro, buildup, drop, breakdown, outro)
    """
    name_lower = cue_name.lower()

    # Check for common EDM section names
    if "intro" in name_lower:
        return "intro"
    elif "build" in name_lower:
        return "buildup"
    elif "drop" in name_lower or "main" in name_lower or "chorus" in name_lower:
        return "drop"
    elif "break" in name_lower or "verse" in name_lower:
        return "breakdown"
    elif "outro" in name_lower or "end" in name_lower:
        return "outro"
    else:
        # Default to breakdown for unrecognized cue names
        # (most neutral section type in EDM)
        return "breakdown"
