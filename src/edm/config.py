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
    """

    detect_bpm: bool = True
    detect_structure: bool = True
    use_madmom: bool = True  # Legacy name - controls beat_this library
    use_librosa: bool = False


class ExternalServicesConfig(BaseModel):
    """Configuration for external services.

    Attributes:
        spotify_client_id: Spotify API client ID.
        spotify_client_secret: Spotify API client secret.
        enable_beatport: Enable Beatport lookups.
        enable_tunebat: Enable TuneBat lookups.
        cache_ttl: Cache time-to-live in seconds.
    """

    spotify_client_id: str | None = Field(None, env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str | None = Field(None, env="SPOTIFY_CLIENT_SECRET")
    enable_beatport: bool = True
    enable_tunebat: bool = True
    cache_ttl: int = 3600  # 1 hour


class EDMConfig(BaseModel):
    """Main configuration for EDM library.

    Attributes:
        analysis: Analysis configuration.
        external_services: External services configuration.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file.
    """

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)
    log_level: str = "INFO"
    log_file: Path | None = None

    class Config:
        env_prefix = "EDM_"


def load_config(config_path: Path | None = None) -> EDMConfig:
    """Load configuration from file.

    **NOTE:** TOML configuration file loading is not yet implemented.
    This function currently only loads configuration from environment
    variables and returns default configuration regardless of file contents.

    Args:
        config_path: Path to TOML configuration file. If not provided, looks for
            config in default location (~/.config/edm/config.toml).

    Returns:
        Loaded and validated configuration (currently always returns default config
        with environment variable overrides only).

    Examples:
        >>> config = load_config()
        >>> print(f"Log level: {config.log_level}")
        Log level: INFO
    """
    if config_path is None:
        config_path = Path.home() / ".config" / "edm" / "config.toml"

    if config_path.exists():
        logger.info("loading config", config_path=str(config_path))
        # TODO: Implement TOML loading with tomli
        # Currently this logs the path but does not actually parse or load the file

    # Load from environment variables only
    return EDMConfig()
