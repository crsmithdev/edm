"""Audio analysis module for EDM tracks."""

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
]
