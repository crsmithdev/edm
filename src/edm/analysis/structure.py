"""Track structure and section detection."""

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import structlog
from mutagen import File as MutagenFile

from edm.analysis.bars import TimeSignature, bar_count_for_range, time_to_bars
from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure_detector import (
    DetectedSection,
    EnergyDetector,
    MSAFDetector,
    get_detector,
    merge_short_sections,
)

logger = structlog.get_logger(__name__)


@dataclass
class Section:
    """A section of a track.

    Attributes:
        label: Section label (e.g., 'intro', 'drop', 'breakdown').
        start_time: Start time in seconds.
        end_time: End time in seconds.
        confidence: Confidence score between 0 and 1.
        start_bar: Start bar position (0-indexed). None if BPM unavailable.
        end_bar: End bar position. None if BPM unavailable.
        bar_count: Number of bars in section. None if BPM unavailable.
    """

    label: str
    start_time: float
    end_time: float
    confidence: float
    start_bar: float | None = None
    end_bar: float | None = None
    bar_count: float | None = None


@dataclass
class RawSection:
    """Raw detected section with full detail for debugging.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        start_bar: Start bar position (fractional).
        end_bar: End bar position (fractional).
        label: Section label from detector.
        confidence: Confidence score between 0 and 1.
    """

    start: float
    end: float
    start_bar: float | None
    end_bar: float | None
    label: str
    confidence: float


@dataclass
class StructureResult:
    """Result of structure analysis.

    Attributes:
        sections: Detected span sections in chronological order.
        events: Detected moment-based events (drops, kicks).
        raw: Raw detected sections with full detail (timestamps, fractional bars, confidence).
        duration: Total track duration in seconds.
        detector: Name of the detector used.
        bpm: BPM used for bar calculations. None if unavailable.
        downbeat: Time of first beat in seconds.
        time_signature: Time signature as (numerator, denominator) tuple.
    """

    sections: list[Section]
    events: list[tuple[int, str]]
    raw: list[RawSection]
    duration: float
    detector: str
    bpm: float | None = None
    downbeat: float | None = None
    time_signature: TimeSignature = (4, 4)


def analyze_structure(
    filepath: Path,
    detector: str = "auto",
    bpm: float | None = None,
    include_bars: bool = True,
    time_signature: TimeSignature = (4, 4),
) -> StructureResult:
    """Analyze the structure of an EDM track.

    Detects sections like intro, buildup, drop, breakdown, and outro using
    MSAF (Music Structure Analysis Framework) or energy-based detection.

    Args:
        filepath: Path to the audio file.
        detector: Detection method ('auto', 'msaf', 'energy').
            - 'auto': Use MSAF if available, otherwise energy-based.
            - 'msaf': Use MSAF boundary detection with energy-based labeling.
            - 'energy': Use pure energy-based detection.
        bpm: Optional BPM for bar calculations. If None and include_bars=True,
            will analyze BPM automatically.
        include_bars: If True, calculate bar positions for sections. Default True.
        time_signature: Time signature for bar calculations. Default (4, 4).

    Returns:
        Detected structure with sections and timing.

    Raises:
        AudioFileError: If the audio file cannot be loaded.
        AnalysisError: If structure detection fails.

    Examples:
        >>> from pathlib import Path
        >>> result = analyze_structure(Path("track.mp3"))
        >>> for section in result.sections:
        ...     print(f"{section.label}: {section.start_time:.1f}s - {section.end_time:.1f}s")
        intro: 0.0s - 32.0s
        buildup: 32.0s - 64.0s
        drop: 64.0s - 128.0s
    """
    logger.debug("analyzing structure", filepath=str(filepath), detector=detector)

    # Get audio duration
    try:
        audio = MutagenFile(filepath)
        if audio is None:
            raise ValueError(f"Unable to read file format: {filepath.suffix}")
        duration = audio.info.length if hasattr(audio.info, "length") else 180.0
    except Exception as e:
        logger.warning(
            "failed to read audio duration, will compute from audio",
            filepath=str(filepath),
            error=str(e),
        )
        duration = None

    # Get detector
    structure_detector = get_detector(detector)

    # Determine detector name for reporting
    if isinstance(structure_detector, MSAFDetector):
        detector_name = "msaf"
    elif isinstance(structure_detector, EnergyDetector):
        detector_name = "energy"
    else:
        detector_name = "unknown"

    # Run detection - errors propagate
    detected_sections = structure_detector.detect(filepath)

    # Get BPM and beat information for bar calculations if requested
    result_bpm = bpm
    result_downbeat = None
    if include_bars and bpm is None:
        try:
            logger.debug("analyzing BPM and beat grid for bar calculations", filepath=str(filepath))
            bpm_result = analyze_bpm(filepath)
            result_bpm = bpm_result.bpm
            logger.debug("bpm detected for bar calculations", bpm=result_bpm)

            # Also get beat grid for downbeat
            from edm.analysis.beat_detector import detect_beats

            beat_grid = detect_beats(filepath)
            result_downbeat = beat_grid.first_beat_time
            logger.debug("downbeat detected", downbeat=result_downbeat)
        except Exception as e:
            logger.debug("bpm/beat analysis failed, bar calculations will be skipped", error=str(e))
            result_bpm = None
            result_downbeat = None

    # Merge short sections (minimum 8 bars)
    detected_sections = merge_short_sections(
        detected_sections, bpm=result_bpm, min_section_bars=8, time_signature=time_signature
    )

    # Convert to Section objects (only for non-event sections)
    sections = [
        Section(
            label=s.label,
            start_time=s.start_time,
            end_time=s.end_time,
            confidence=s.confidence,
        )
        for s in detected_sections
        if not s.is_event
    ]

    # Post-process sections
    sections = _post_process_sections(sections, duration)

    # Merge consecutive sections with same label
    sections = _merge_consecutive_same_label(sections)

    # Add bar calculations if requested and BPM available
    if include_bars and result_bpm is not None:
        sections = _add_bar_calculations(
            sections, result_bpm, time_signature, result_downbeat or 0.0
        )

    # Format events with bar numbers
    events: list[tuple[int, str]] = []
    if include_bars and result_bpm is not None and result_downbeat is not None:
        event_sections = [s for s in detected_sections if s.is_event]
        _, events = _format_structure_output(
            event_sections, result_bpm, time_signature, result_downbeat
        )

    # Build raw sections with fractional bar positions
    # Musical convention: if section ends at bar B, next starts at B+1
    raw_sections: list[RawSection] = []
    for i, s in enumerate(detected_sections):
        start_bar: float | None = None
        end_bar: float | None = None
        if result_bpm is not None and result_downbeat is not None:
            start_result = time_to_bars(s.start_time, result_bpm, time_signature, result_downbeat)
            if start_result:
                start_bar_num, start_frac_beat = start_result
                # Calculate full fractional bar value
                start_bar = start_bar_num + (start_frac_beat / time_signature[0])

            end_result = time_to_bars(s.end_time, result_bpm, time_signature, result_downbeat)
            if end_result:
                end_bar_num, end_frac_beat = end_result
                # Calculate full fractional bar value
                end_bar = end_bar_num + (end_frac_beat / time_signature[0])

        raw_sections.append(
            RawSection(
                start=round(s.start_time, 2),
                end=round(s.end_time, 2),
                start_bar=start_bar,
                end_bar=end_bar,
                label=s.label,
                confidence=round(s.confidence, 2),
            )
        )

    # Update duration from detection if not available
    if duration is None and sections:
        duration = sections[-1].end_time

    if duration is None:
        duration = 180.0  # Fallback

    logger.info(
        "structure analysis complete",
        filepath=str(filepath),
        detector=detector_name,
        sections=len(sections),
        events=len(events),
        raw=len(raw_sections),
        duration=duration,
        bpm=result_bpm,
        downbeat=result_downbeat,
    )

    return StructureResult(
        sections=sections,
        events=events,
        raw=raw_sections,
        duration=duration,
        detector=detector_name,
        bpm=result_bpm,
        downbeat=result_downbeat,
        time_signature=time_signature,
    )


def _post_process_sections(sections: list[Section], duration: float | None) -> list[Section]:
    """Post-process sections to ensure validity.

    Ensures:
    - Sections are sorted by start time
    - No overlaps between sections
    - Full track coverage at start/end (if duration known)
    - Gaps between sections are allowed (interpreted as unclassified regions)

    Args:
        sections: List of detected sections.
        duration: Total track duration (optional).

    Returns:
        Post-processed sections.
    """
    if not sections:
        # Return minimal structure if no sections detected
        if duration:
            return [Section(label="intro", start_time=0.0, end_time=duration, confidence=0.5)]
        return []

    # Sort by start time
    sections = sorted(sections, key=lambda s: s.start_time)

    # Ensure first section starts at 0
    if sections[0].start_time > 0.1:
        sections.insert(
            0,
            Section(
                label="intro",
                start_time=0.0,
                end_time=sections[0].start_time,
                confidence=0.7,
            ),
        )

    # Remove overlaps and gaps
    processed = [sections[0]]
    for section in sections[1:]:
        prev = processed[-1]

        # Handle overlap
        if section.start_time < prev.end_time:
            # Adjust start time to end of previous
            section = Section(
                label=section.label,
                start_time=prev.end_time,
                end_time=max(section.end_time, prev.end_time + 1.0),
                confidence=section.confidence,
            )

        # Gaps are allowed - don't extend previous section

        # Skip zero-duration sections
        if section.end_time <= section.start_time:
            continue

        processed.append(section)

    # Ensure last section extends to duration
    if duration and processed:
        last = processed[-1]
        if last.end_time < duration - 0.1:
            processed[-1] = Section(
                label=last.label,
                start_time=last.start_time,
                end_time=duration,
                confidence=last.confidence,
            )

    return processed


def _format_structure_output(
    detected_sections: list[DetectedSection],
    bpm: float,
    time_signature: TimeSignature,
    downbeat: float,
) -> tuple[list[tuple[int, int, str]], list[tuple[int, str]]]:
    """Format detected sections into spans and events with 1-indexed bars.

    Args:
        detected_sections: List of detected sections from detector.
        bpm: BPM for bar calculations.
        time_signature: Time signature tuple.
        downbeat: Time of first downbeat in seconds.

    Returns:
        Tuple of (spans, events) where:
        - spans: List of [start_bar, end_bar, label] (1-indexed, inclusive)
        - events: List of [bar, label] (1-indexed)
    """
    spans = []
    events = []

    for section in detected_sections:
        if section.is_event:
            # Event: single bar number
            bar_result = time_to_bars(section.start_time, bpm, time_signature, downbeat)
            if bar_result is None:
                continue
            bar_num, _ = bar_result
            bar_1indexed = int(ceil(bar_num)) + 1
            events.append((bar_1indexed, section.label))
        else:
            # Span: start and end bars
            start_bar_result = time_to_bars(section.start_time, bpm, time_signature, downbeat)
            end_bar_result = time_to_bars(section.end_time, bpm, time_signature, downbeat)
            if start_bar_result is None or end_bar_result is None:
                continue
            start_bar_num, _ = start_bar_result
            end_bar_num, _ = end_bar_result

            start_bar_1indexed = int(ceil(start_bar_num)) + 1
            end_bar_1indexed = int(floor(end_bar_num))

            # Ensure start <= end
            if end_bar_1indexed < start_bar_1indexed:
                end_bar_1indexed = start_bar_1indexed

            spans.append((start_bar_1indexed, end_bar_1indexed, section.label))

    return spans, events


def _merge_consecutive_same_label(sections: list[Section]) -> list[Section]:
    """Merge consecutive sections with the same label.

    Args:
        sections: List of sections.

    Returns:
        List with consecutive same-label sections merged.
    """
    if not sections:
        return []

    merged = [sections[0]]

    for section in sections[1:]:
        prev = merged[-1]

        if section.label == prev.label:
            # Merge: extend previous section's end time
            merged[-1] = Section(
                label=prev.label,
                start_time=prev.start_time,
                end_time=section.end_time,
                confidence=max(prev.confidence, section.confidence),
                start_bar=prev.start_bar,
                end_bar=section.end_bar,
                bar_count=None,  # Recalculate if needed
            )
        else:
            merged.append(section)

    return merged


def _add_bar_calculations(
    sections: list[Section], bpm: float, time_signature: TimeSignature, downbeat: float = 0.0
) -> list[Section]:
    """Add bar position calculations to sections.

    Args:
        sections: List of sections with time positions.
        bpm: BPM for bar calculations.
        time_signature: Time signature for bar calculations.
        downbeat: Time in seconds where bar 1 begins. Default 0.0.

    Returns:
        Sections with bar fields populated.
    """
    result = []
    for section in sections:
        # Calculate bar positions
        start_bars = time_to_bars(section.start_time, bpm, time_signature, downbeat)
        end_bars = time_to_bars(section.end_time, bpm, time_signature, downbeat)

        if start_bars is not None and end_bars is not None:
            start_bar = start_bars[0] + (start_bars[1] / time_signature[0])
            end_bar = end_bars[0] + (end_bars[1] / time_signature[0])
            bar_count_val = bar_count_for_range(
                section.start_time, section.end_time, bpm, time_signature
            )
        else:
            start_bar = None
            end_bar = None
            bar_count_val = None

        result.append(
            Section(
                label=section.label,
                start_time=section.start_time,
                end_time=section.end_time,
                confidence=section.confidence,
                start_bar=start_bar,
                end_bar=end_bar,
                bar_count=bar_count_val,
            )
        )

    return result
