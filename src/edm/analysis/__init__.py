"""Audio analysis module for EDM tracks."""

from edm.analysis.bars import (
    TimeSignature,
    bar_count_for_range,
    bars_to_time,
    check_bar_alignment,
    get_section_at_bar,
    time_to_bars,
)
from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure import Section, StructureResult, analyze_structure
from edm.analysis.structure_detector import (
    DetectedSection,
    EnergyDetector,
    MSAFDetector,
    StructureDetector,
    get_detector,
)

__all__ = [
    "analyze_bpm",
    "analyze_structure",
    "Section",
    "StructureResult",
    "StructureDetector",
    "DetectedSection",
    "MSAFDetector",
    "EnergyDetector",
    "get_detector",
    "TimeSignature",
    "time_to_bars",
    "bars_to_time",
    "bar_count_for_range",
    "get_section_at_bar",
    "check_bar_alignment",
]
