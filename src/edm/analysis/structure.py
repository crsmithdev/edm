"""Track structure and section detection."""

from dataclasses import dataclass
from pathlib import Path

import structlog
from mutagen import File as MutagenFile

from edm.analysis.bars import TimeSignature, bar_count_for_range, time_to_bars
from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure_detector import (
    EnergyDetector,
    MSAFDetector,
    get_detector,
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
class StructureResult:
    """Result of structure analysis.

    Attributes:
        sections: Detected sections in chronological order.
        duration: Total track duration in seconds.
        detector: Name of the detector used.
        bpm: BPM used for bar calculations. None if unavailable.
    """

    sections: list[Section]
    duration: float
    detector: str
    bpm: float | None = None


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
    if structure_detector is None:
        logger.warning("no detector available, using energy fallback")
        structure_detector = EnergyDetector()

    # Determine detector name for reporting
    if isinstance(structure_detector, MSAFDetector):
        detector_name = "msaf"
    elif isinstance(structure_detector, EnergyDetector):
        detector_name = "energy"
    else:
        detector_name = "unknown"

    # Run detection
    try:
        detected_sections = structure_detector.detect(filepath)
    except Exception as e:
        logger.error("structure detection failed", error=str(e))
        # Fall back to energy detector if MSAF fails
        if detector_name == "msaf":
            logger.info("falling back to energy detector")
            structure_detector = EnergyDetector()
            detector_name = "energy"
            detected_sections = structure_detector.detect(filepath)
        else:
            raise

    # Get BPM for bar calculations if requested
    result_bpm = bpm
    if include_bars and bpm is None:
        try:
            logger.debug("analyzing BPM for bar calculations", filepath=str(filepath))
            bpm_result = analyze_bpm(filepath)
            result_bpm = bpm_result.bpm
            logger.debug("bpm detected for bar calculations", bpm=result_bpm)
        except Exception as e:
            logger.debug("bpm analysis failed, bar calculations will be skipped", error=str(e))
            result_bpm = None

    # Convert to Section objects
    sections = [
        Section(
            label=s.label,
            start_time=s.start_time,
            end_time=s.end_time,
            confidence=s.confidence,
        )
        for s in detected_sections
    ]

    # Post-process sections
    sections = _post_process_sections(sections, duration)

    # Add bar calculations if requested and BPM available
    if include_bars and result_bpm is not None:
        sections = _add_bar_calculations(sections, result_bpm, time_signature)

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
        duration=duration,
    )

    return StructureResult(
        sections=sections,
        duration=duration,
        detector=detector_name,
        bpm=result_bpm,
    )


def _post_process_sections(sections: list[Section], duration: float | None) -> list[Section]:
    """Post-process sections to ensure validity.

    Ensures:
    - Sections are sorted by start time
    - No overlaps between sections
    - Full track coverage (if duration known)
    - No gaps between sections

    Args:
        sections: List of detected sections.
        duration: Total track duration (optional).

    Returns:
        Post-processed sections.
    """
    if not sections:
        # Return minimal structure if no sections detected
        if duration:
            return [
                Section(label="intro", start_time=0.0, end_time=duration, confidence=0.5)
            ]
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

        # Handle gap
        elif section.start_time > prev.end_time + 0.1:
            # Extend previous section to fill gap
            processed[-1] = Section(
                label=prev.label,
                start_time=prev.start_time,
                end_time=section.start_time,
                confidence=prev.confidence,
            )

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


def _add_bar_calculations(
    sections: list[Section], bpm: float, time_signature: TimeSignature
) -> list[Section]:
    """Add bar position calculations to sections.

    Args:
        sections: List of sections with time positions.
        bpm: BPM for bar calculations.
        time_signature: Time signature for bar calculations.

    Returns:
        Sections with bar fields populated.
    """
    result = []
    for section in sections:
        # Calculate bar positions
        start_bars = time_to_bars(section.start_time, bpm, time_signature)
        end_bars = time_to_bars(section.end_time, bpm, time_signature)

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
