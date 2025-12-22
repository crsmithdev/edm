"""Annotation loading and saving service."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from edm.data.metadata import AnnotationMetadata
from edm.data.schema import Annotation, AudioMetadata, StructureSection

from .audio_service import AudioService


class AnnotationService:
    """Handles annotation CRUD operations."""

    def __init__(self, config: Any, audio_service: AudioService):
        """Initialize annotation service.

        Args:
            config: Flask configuration object with annotation directories
            audio_service: AudioService instance for audio metadata
        """
        self.annotation_dir = config["ANNOTATION_DIR"]
        self.reference_dir = self.annotation_dir / "reference"
        self.generated_dir = self.annotation_dir / "generated"
        self.valid_labels = config["VALID_LABELS"]
        self.audio_service = audio_service

    def find_annotation_for_file(self, filename: str) -> tuple[Path | None, int | None]:
        """Find annotation YAML file for given audio filename.

        Args:
            filename: Audio filename (e.g., "Artist - Track.flac")

        Returns:
            (annotation_path, tier) or (None, None) if not found
            Prefers reference (tier 1) over generated (tier 2)
        """
        stem = Path(filename).stem

        # Check reference first (tier 1)
        ref_path = self.reference_dir / f"{stem}.yaml"
        if ref_path.exists():
            return (ref_path, 1)

        # Fall back to generated (tier 2)
        gen_path = self.generated_dir / f"{stem}.yaml"
        if gen_path.exists():
            return (gen_path, 2)

        # Try case-insensitive search in both directories
        for directory, tier in [
            (self.reference_dir, 1),
            (self.generated_dir, 2),
        ]:
            if directory.exists():
                for yaml_file in directory.glob("*.yaml"):
                    if yaml_file.stem.lower() == stem.lower():
                        return (yaml_file, tier)

        return (None, None)

    def load_annotation(self, filename: str) -> dict[str, Any] | None:
        """Load annotation data for a track if it exists.

        Prefers reference (hand-tagged) annotations over generated annotations.

        Args:
            filename: Audio filename

        Returns:
            Dictionary with bpm, downbeat, boundaries, and tier if found, None otherwise
        """
        annotation_path, tier = self.find_annotation_for_file(filename)

        if not annotation_path or not annotation_path.exists():
            return None

        try:
            with open(annotation_path) as f:
                annotation_data = yaml.safe_load(f)

            if not annotation_data:
                return None

            # Audio section is required for valid annotation
            if "audio" not in annotation_data:
                return None

            result = {"tier": tier}

            # Load audio metadata
            result["bpm"] = annotation_data["audio"].get("bpm")
            result["downbeat"] = annotation_data["audio"].get("downbeat", 0.0)

            # Load structure boundaries
            if "structure" in annotation_data:
                boundaries = []
                for section in annotation_data["structure"]:
                    if isinstance(section, dict):
                        boundaries.append(
                            {
                                "time": section.get("time", 0.0),
                                "label": section.get("label", "unlabeled"),
                            }
                        )
                result["boundaries"] = boundaries

            return result
        except Exception as e:
            print(f"Warning: Could not load annotation: {e}")

        return None

    def load_generated_annotation(self, filename: str) -> dict[str, Any] | None:
        """Load full annotation data from generated tier only.

        Args:
            filename: Audio filename

        Returns:
            Dictionary with bpm, downbeat, and boundaries if found, None otherwise
        """
        stem = Path(filename).stem
        gen_path = self.generated_dir / f"{stem}.yaml"

        if not gen_path.exists():
            # Try case-insensitive search
            if self.generated_dir.exists():
                for yaml_file in self.generated_dir.glob("*.yaml"):
                    if yaml_file.stem.lower() == stem.lower():
                        gen_path = yaml_file
                        break
                else:
                    return None
            else:
                return None

        try:
            with open(gen_path) as f:
                annotation_data = yaml.safe_load(f)

            if not annotation_data:
                return None

            result = {}

            # Load audio metadata
            if "audio" in annotation_data:
                result["bpm"] = annotation_data["audio"].get("bpm")
                result["downbeat"] = annotation_data["audio"].get("downbeat", 0.0)

            # Load structure boundaries
            if "structure" in annotation_data:
                boundaries = []
                for section in annotation_data["structure"]:
                    if isinstance(section, dict):
                        label = section.get("label", "default")
                        # Migrate old labels to new names
                        if label == "unlabeled":
                            label = "default"
                        elif label == "breakbuild":
                            label = "breakdown-buildup"
                        boundaries.append(
                            {
                                "time": section.get("time", 0.0),
                                "label": label,
                            }
                        )
                result["boundaries"] = boundaries

            return result if result else None
        except Exception as e:
            print(f"Warning: Could not load generated annotation: {e}")

        return None

    def save_annotation(
        self, filename: str, bpm: float, downbeat: float, boundaries: list[dict]
    ) -> Path:
        """Save annotation to YAML file.

        Args:
            filename: Audio filename
            bpm: Track BPM
            downbeat: Downbeat time in seconds
            boundaries: List of boundary dictionaries with 'time' and 'label'

        Returns:
            Path to saved annotation file

        Raises:
            ValueError: If boundaries contain invalid labels
        """
        # Validate labels
        for boundary in boundaries:
            label = boundary.get("label")
            if label and label not in self.valid_labels:
                raise ValueError(f"Invalid label: {label}")

        # Get audio metadata
        audio_path = self.audio_service.validate_audio_path(filename)
        duration = self.audio_service.get_duration(filename)

        # Convert timestamps to bars and create StructureSection objects
        structure_sections = []
        for boundary in boundaries:
            time = boundary["time"]
            label = boundary["label"]

            # Calculate bar number (1-indexed)
            bar = self._time_to_bar(time, bpm, downbeat)

            structure_sections.append(
                StructureSection(bar=bar, label=label, time=time, confidence=1.0)
            )

        # Sort by time
        structure_sections = sorted(structure_sections, key=lambda x: x.time)

        # Build Annotation object using edm schema
        now = datetime.now(UTC)
        annotation = Annotation(
            metadata=AnnotationMetadata(
                tier=1,  # Tier 1 = manual annotation
                confidence=1.0,
                source="manual",
                created=now,
                modified=now,
                annotator="web_tool",
                flags=[],
            ),
            audio=AudioMetadata(
                file=audio_path,
                duration=duration,
                bpm=bpm,
                downbeat=downbeat,
                time_signature=(4, 4),
            ),
            structure=structure_sections,
        )

        # Save to reference directory
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.reference_dir / f"{Path(filename).stem}.yaml"
        annotation.to_yaml(output_file)

        return output_file

    def _time_to_bar(self, time: float, bpm: float, downbeat: float) -> int:
        """Convert time to bar number.

        Args:
            time: Time in seconds
            bpm: BPM
            downbeat: Downbeat time in seconds

        Returns:
            Bar number (1-indexed)
        """
        bar_duration = 60.0 / bpm * 4.0  # Duration of one bar (4/4 time)
        bar = int((time - downbeat) / bar_duration) + 1
        return max(1, bar)  # Minimum bar 1
