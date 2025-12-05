"""Annotation metadata handling for data management."""

from datetime import datetime
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnnotationTier(IntEnum):
    """Quality tier for annotations.

    Attributes:
        VERIFIED: Tier 1 - Manually verified by human annotator
        AUTO_CLEANED: Tier 2 - Automatically cleaned/refined from source
        AUTO_GENERATED: Tier 3 - Automatically generated, needs review
    """

    VERIFIED = 1
    AUTO_CLEANED = 2
    AUTO_GENERATED = 3


class ValidationStatus(BaseModel):
    """Validation checks performed on annotation."""

    beat_grid_valid: bool = Field(description="Beat grid aligns with audio timing")
    cue_points_snapped: bool = Field(description="Cue points are snapped to beat grid")
    min_section_length: bool = Field(description="All sections meet minimum length requirement")


class AnnotationMetadata(BaseModel):
    """Metadata about an annotation's provenance and quality.

    Tracks where annotations come from, their quality tier,
    confidence scores, and validation status.
    """

    tier: AnnotationTier = Field(
        description="Quality tier: 1=verified, 2=auto-cleaned, 3=auto-generated"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score for this annotation [0-1]"
    )
    source: str = Field(
        description="Annotation source: 'rekordbox', 'msaf', 'ml_model_v1', 'manual'"
    )
    created: datetime = Field(
        default_factory=datetime.now, description="Timestamp when annotation was created"
    )
    modified: datetime = Field(
        default_factory=datetime.now, description="Timestamp when annotation was last modified"
    )
    verified_by: Optional[str] = Field(
        default=None, description="User who verified this annotation (if tier=1)"
    )
    notes: Optional[str] = Field(
        default=None, description="Human-readable notes about this annotation"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Flags for review: 'needs_review', 'boundary_uncertain', etc.",
    )
    validation: ValidationStatus = Field(
        default_factory=lambda: ValidationStatus(
            beat_grid_valid=False, cue_points_snapped=False, min_section_length=False
        ),
        description="Validation checks performed",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate that source is from known list."""
        valid_sources = {"rekordbox", "msaf", "manual", "energy"}
        # Also allow ML model versions like ml_model_v1, ml_model_v2
        if v not in valid_sources and not v.startswith("ml_model_"):
            raise ValueError(
                f"Invalid source '{v}'. Must be one of {valid_sources} or start with 'ml_model_'"
            )
        return v

    @field_validator("verified_by")
    @classmethod
    def validate_verified_by(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure verified_by is set for tier 1 annotations."""
        # Access tier from the model being validated
        tier = info.data.get("tier")
        if tier == AnnotationTier.VERIFIED and not v:
            raise ValueError("verified_by must be set for tier 1 (verified) annotations")
        return v
