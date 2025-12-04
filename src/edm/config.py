"""Configuration management."""

from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AnalysisConfig(BaseModel):
    """Configuration for track analysis.

    Attributes:
        detect_bpm: Enable BPM detection.
        detect_structure: Enable structure detection.
        use_madmom: Use beat_this for BPM detection (legacy parameter name).
        use_librosa: Use librosa for BPM detection.
        structure_detector: Structure detector type ('auto'/'msaf', 'energy').
    """

    detect_bpm: bool = True
    detect_structure: bool = True
    use_madmom: bool = True  # Legacy name - controls beat_this library
    use_librosa: bool = False
    structure_detector: str = "auto"  # 'auto'/'msaf' (required), 'energy' (explicit fallback)


class EDMConfig(BaseModel):
    """Main configuration for EDM library.

    Attributes:
        analysis: Analysis configuration.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file.
        bpm_lookup_strategy: Order of BPM lookup sources.
    """

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    log_level: str = "INFO"
    log_file: Path | None = None
    bpm_lookup_strategy: list[str] = Field(
        default=["metadata", "computed"],
        description="Order of BPM lookup sources: metadata, computed",
    )

    class Config:
        env_prefix = "EDM_"


def load_config(config_path: Path | None = None) -> EDMConfig:
    """Load configuration.

    Args:
        config_path: Unused, kept for backward compatibility.

    Returns:
        Configuration loaded from environment variables.

    Examples:
        >>> config = load_config()
        >>> print(f"Log level: {config.log_level}")
        Log level: INFO
    """
    return EDMConfig()
