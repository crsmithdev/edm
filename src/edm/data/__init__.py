"""Data management module for EDM annotations and datasets.

This module provides:
- Annotation schema validation
- Metadata handling (tiers, confidence scores, provenance)
- Export utilities for ML frameworks
- Dataset versioning support via DVC
"""

from edm.data.metadata import AnnotationMetadata, AnnotationTier
from edm.data.schema import Annotation, AudioMetadata, StructureSection

__all__ = [
    "Annotation",
    "AnnotationMetadata",
    "AnnotationTier",
    "AudioMetadata",
    "StructureSection",
]
