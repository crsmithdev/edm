"""Annotation schema definitions using Pydantic.

Important Conventions:
    - Bar numbers are 1-indexed (bar 1 is the first bar, not bar 0)
    - Times are in seconds (not M:SS.mm format)
    - Confidence scores are in range [0.0, 1.0]
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from edm.data.metadata import AnnotationMetadata


class AudioMetadata(BaseModel):
    """Audio file metadata.

    Attributes:
        file: Path to audio file
        duration: Track duration in seconds (not M:SS.mm format)
        bpm: Beats per minute
        downbeat: First downbeat timestamp in seconds
        time_signature: [numerator, denominator], e.g., [4, 4] for 4/4 time
        key: Optional key in Camelot notation (e.g., "8A")
    """

    file: Path = Field(description="Path to audio file")
    duration: float = Field(gt=0, description="Track duration in seconds")
    bpm: float = Field(gt=0, description="Beats per minute")
    downbeat: float = Field(ge=0, description="First downbeat timestamp in seconds")
    time_signature: tuple[int, int] = Field(
        default=(4, 4), description="Time signature as [numerator, denominator]"
    )
    key: Optional[str] = Field(
        default=None, description="Track key in Camelot notation (e.g., '8A')"
    )

    @field_validator("time_signature", mode="before")
    @classmethod
    def validate_time_signature(cls, v: Any) -> tuple[int, int]:
        """Convert list to tuple and validate."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        raise ValueError("time_signature must be [numerator, denominator]")


class StructureSection(BaseModel):
    """A single structure section boundary.

    Attributes:
        bar: Bar number (1-indexed)
        label: Section label (intro, buildup, drop, breakdown, outro)
        time: Timestamp in seconds
        confidence: Confidence score for this boundary [0-1]
    """

    bar: int = Field(ge=1, description="Bar number (1-indexed)")
    label: str = Field(description="Section label")
    time: float = Field(ge=0, description="Timestamp in seconds")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Confidence score [0-1]")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate section label, warn on unknown values."""
        import warnings

        valid_labels = {"intro", "buildup", "drop", "breakdown", "breakbuild", "outro", "unlabeled"}
        if v not in valid_labels:
            warnings.warn(
                f"Unknown label '{v}' (expected one of {valid_labels}). "
                f"Will be mapped to 'unlabeled' during training.",
                UserWarning,
                stacklevel=2,
            )
        return v


class EnergySectionData(BaseModel):
    """Energy data for a structure section.

    Attributes:
        section: Index into structure array
        bass: Bass energy [0-1]
        mid: Mid-range energy [0-1]
        high: High frequency energy [0-1]
    """

    section: int = Field(ge=0, description="Index into structure array")
    bass: float = Field(ge=0.0, le=1.0, description="Bass energy [0-1]")
    mid: float = Field(ge=0.0, le=1.0, description="Mid-range energy [0-1]")
    high: float = Field(ge=0.0, le=1.0, description="High frequency energy [0-1]")


class EnergyBoundaryData(BaseModel):
    """Energy change at a structure boundary.

    Attributes:
        boundary: Index into structure array
        delta: Energy change magnitude at this boundary [-1, 1]
    """

    boundary: int = Field(ge=0, description="Index into structure array")
    delta: float = Field(ge=-1.0, le=1.0, description="Energy change at boundary [-1, 1]")


class EnergyData(BaseModel):
    """Overall energy analysis data.

    Attributes:
        overall: Overall track energy [0-1]
        by_section: Energy breakdown per section
        at_boundaries: Energy changes at boundaries
    """

    overall: float = Field(ge=0.0, le=1.0, description="Overall track energy [0-1]")
    by_section: list[EnergySectionData] = Field(
        default_factory=list, description="Energy per section"
    )
    at_boundaries: list[EnergyBoundaryData] = Field(
        default_factory=list, description="Energy changes at boundaries"
    )


class Annotation(BaseModel):
    """Complete annotation for an EDM track.

    This is the top-level schema that includes metadata, audio info,
    structure annotations, and optional energy data.
    """

    metadata: AnnotationMetadata = Field(
        description="Annotation metadata (tier, confidence, source)"
    )
    audio: AudioMetadata = Field(description="Audio file metadata")
    structure: list[StructureSection] = Field(description="Structure section boundaries")
    energy: Optional[EnergyData] = Field(default=None, description="Optional energy analysis data")

    @classmethod
    def from_yaml(cls, path: Path) -> "Annotation":
        """Load annotation from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Annotation object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't match schema
        """
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save annotation to YAML file.

        Args:
            path: Path to save YAML file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling special types
        data = self.model_dump(mode="json", exclude_none=True)

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def validate_structure(self) -> list[str]:
        """Validate structure annotations for common issues.

        Returns:
            List of validation warning messages (empty if all valid)
        """
        warnings = []

        if len(self.structure) < 2:
            warnings.append("Structure has fewer than 2 sections")

        # Check for bar ordering
        for i in range(len(self.structure) - 1):
            if self.structure[i].bar >= self.structure[i + 1].bar:
                warnings.append(
                    f"Bars not in ascending order: "
                    f"section {i} bar {self.structure[i].bar} >= "
                    f"section {i + 1} bar {self.structure[i + 1].bar}"
                )

        # Check for time ordering
        for i in range(len(self.structure) - 1):
            if self.structure[i].time >= self.structure[i + 1].time:
                warnings.append(
                    f"Times not in ascending order: "
                    f"section {i} time {self.structure[i].time} >= "
                    f"section {i + 1} time {self.structure[i + 1].time}"
                )

        # Check for section length (minimum 4 bars for EDM)
        for i in range(len(self.structure) - 1):
            bar_length = self.structure[i + 1].bar - self.structure[i].bar
            if bar_length < 4:
                warnings.append(
                    f"Section {i} ({self.structure[i].label}) is only "
                    f"{bar_length} bars (minimum 4 recommended)"
                )

        return warnings
