"""Validation utilities for annotation data quality."""

from pathlib import Path

from edm.data.rekordbox import RekordboxTrack
from edm.data.schema import Annotation


class ValidationIssue:
    """Represents a validation issue found in annotation data."""

    def __init__(self, severity: str, message: str, field: str | None = None):
        """Initialize validation issue.

        Args:
            severity: 'error' or 'warning'
            message: Description of the issue
            field: Optional field name where issue was found
        """
        self.severity = severity
        self.message = message
        self.field = field

    def __repr__(self) -> str:
        field_str = f" [{self.field}]" if self.field else ""
        return f"{self.severity.upper()}{field_str}: {self.message}"


class ValidationResult:
    """Results from validating annotation data."""

    def __init__(self):
        self.issues: list[ValidationIssue] = []

    def add_error(self, message: str, field: str | None = None) -> None:
        """Add an error-level issue."""
        self.issues.append(ValidationIssue("error", message, field))

    def add_warning(self, message: str, field: str | None = None) -> None:
        """Add a warning-level issue."""
        self.issues.append(ValidationIssue("warning", message, field))

    @property
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not self.has_errors

    def __repr__(self) -> str:
        if not self.issues:
            return "ValidationResult(valid, no issues)"
        error_count = sum(1 for i in self.issues if i.severity == "error")
        warning_count = sum(1 for i in self.issues if i.severity == "warning")
        return f"ValidationResult({error_count} errors, {warning_count} warnings)"


def validate_rekordbox_track(
    track: RekordboxTrack,
    beat_snap_tolerance_ms: float = 50.0,
    min_duration: float = 60.0,
    max_duration: float = 900.0,
) -> ValidationResult:
    """Validate Rekordbox track data quality.

    Args:
        track: Rekordbox track to validate
        beat_snap_tolerance_ms: Maximum ms offset for cue points to be "on beat"
        min_duration: Minimum track duration in seconds
        max_duration: Maximum track duration in seconds

    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()

    # Check audio file exists
    if not track.location.exists():
        result.add_error(f"Audio file not found: {track.location}", "location")

    # Check duration sanity
    if track.duration < min_duration:
        result.add_warning(
            f"Track very short: {track.duration:.1f}s (minimum {min_duration}s)",
            "duration",
        )
    if track.duration > max_duration:
        result.add_warning(
            f"Track very long: {track.duration:.1f}s (maximum {max_duration}s)",
            "duration",
        )

    # Check BPM range
    if track.bpm < 80 or track.bpm > 200:
        result.add_warning(
            f"BPM outside typical EDM range: {track.bpm} (expected 80-200)",
            "bpm",
        )

    # Check for cue points
    if not track.cue_points:
        result.add_error("No cue points found", "cue_points")
        return result  # Can't do further validation

    # Check for beat grid
    if not track.beat_grid:
        result.add_warning("No beat grid found, generated from BPM", "beat_grid")
    else:
        # Check beat grid sanity
        if len(track.beat_grid) < 10:
            result.add_warning(
                f"Very few beats in grid: {len(track.beat_grid)}",
                "beat_grid",
            )

    # Check cue point beat snapping
    tolerance_sec = beat_snap_tolerance_ms / 1000.0
    for cue in track.cue_points:
        if track.beat_grid:
            nearest_beat_dist = min(abs(cue.time - beat) for beat in track.beat_grid)
            if nearest_beat_dist > tolerance_sec:
                result.add_warning(
                    f"Cue '{cue.name}' at {cue.time:.2f}s not snapped to beat "
                    f"(nearest: {nearest_beat_dist * 1000:.1f}ms away)",
                    "cue_points",
                )

        # Check cue is within track duration
        if cue.time > track.duration:
            result.add_error(
                f"Cue '{cue.name}' at {cue.time:.2f}s exceeds track duration {track.duration:.2f}s",
                "cue_points",
            )

    # Check cue points are in chronological order
    cue_times = [cue.time for cue in track.cue_points]
    if cue_times != sorted(cue_times):
        result.add_warning("Cue points not in chronological order", "cue_points")

    return result


def validate_annotation(
    annotation: Annotation,
    min_section_bars: int = 4,
    max_section_bars: int = 64,
) -> ValidationResult:
    """Validate Annotation object for data quality.

    Args:
        annotation: Annotation to validate
        min_section_bars: Minimum bars per section
        max_section_bars: Maximum bars per section

    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()

    # Check audio file exists
    if not annotation.audio.file.exists():
        result.add_error(f"Audio file not found: {annotation.audio.file}", "audio.file")

    # Check structure
    if len(annotation.structure) < 2:
        result.add_error(
            f"Too few structure sections: {len(annotation.structure)} (minimum 2)",
            "structure",
        )

    # Check section ordering and lengths
    for i in range(len(annotation.structure) - 1):
        curr = annotation.structure[i]
        next_sec = annotation.structure[i + 1]

        # Check bar ordering
        if curr.bar >= next_sec.bar:
            result.add_error(
                f"Sections not in bar order: section {i} bar {curr.bar} >= section {i + 1} bar {next_sec.bar}",
                "structure",
            )

        # Check time ordering
        if curr.time >= next_sec.time:
            result.add_error(
                f"Sections not in time order: section {i} time {curr.time} >= section {i + 1} time {next_sec.time}",
                "structure",
            )

        # Check section length
        bar_length = next_sec.bar - curr.bar
        if bar_length < min_section_bars:
            result.add_warning(
                f"Section {i} ({curr.label}) very short: {bar_length} bars (minimum {min_section_bars})",
                "structure",
            )
        if bar_length > max_section_bars:
            result.add_warning(
                f"Section {i} ({curr.label}) very long: {bar_length} bars (maximum {max_section_bars})",
                "structure",
            )

    # Check confidence scores
    if annotation.metadata.confidence < 0.5:
        result.add_warning(
            f"Low overall confidence: {annotation.metadata.confidence:.2f}",
            "metadata.confidence",
        )

    for i, section in enumerate(annotation.structure):
        if section.confidence < 0.5:
            result.add_warning(
                f"Section {i} ({section.label}) has low confidence: {section.confidence:.2f}",
                "structure",
            )

    return result


def batch_validate_annotations(
    annotation_dir: Path,
    require_audio_files: bool = False,
) -> dict[str, ValidationResult]:
    """Validate all annotations in a directory.

    Args:
        annotation_dir: Directory containing annotation YAML files
        require_audio_files: If True, error if audio files don't exist

    Returns:
        Dict mapping filename to ValidationResult
    """
    results = {}

    for yaml_file in annotation_dir.rglob("*.yaml"):
        try:
            annotation = Annotation.from_yaml(yaml_file)
            result = validate_annotation(annotation)

            # Optionally downgrade missing audio file from error to warning
            if not require_audio_files:
                for issue in result.issues:
                    if issue.field == "audio.file" and issue.severity == "error":
                        issue.severity = "warning"

            results[yaml_file.name] = result

        except Exception as e:
            result = ValidationResult()
            result.add_error(f"Failed to load annotation: {e}")
            results[yaml_file.name] = result

    return results
