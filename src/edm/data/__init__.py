"""Data management module for EDM annotations and datasets.

This module provides:
- Annotation schema validation
- Metadata handling (tiers, confidence scores, provenance)
- Export utilities for ML frameworks
- Dataset versioning support via DVC
- Rekordbox XML parsing and conversion
"""

from edm.data.converters import batch_convert_rekordbox_xml, rekordbox_to_annotation
from edm.data.metadata import AnnotationMetadata, AnnotationTier
from edm.data.rekordbox import RekordboxTrack, parse_rekordbox_xml
from edm.data.schema import Annotation, AudioMetadata, StructureSection

__all__ = [
    "Annotation",
    "AnnotationMetadata",
    "AnnotationTier",
    "AudioMetadata",
    "StructureSection",
    "RekordboxTrack",
    "parse_rekordbox_xml",
    "rekordbox_to_annotation",
    "batch_convert_rekordbox_xml",
]
