"""Configuration management."""

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalysisConfig(BaseModel):
    """Configuration for track analysis.

    Attributes
    ----------
    detect_bpm : bool
        Enable BPM detection.
    detect_structure : bool
        Enable structure detection.
    use_madmom : bool
        Use madmom for BPM detection.
    use_librosa : bool
        Use librosa for BPM detection.
    """
    detect_bpm: bool = True
    detect_structure: bool = True
    use_madmom: bool = True
    use_librosa: bool = False


class ExternalServicesConfig(BaseModel):
    """Configuration for external services.

    Attributes
    ----------
    spotify_client_id : Optional[str]
        Spotify API client ID.
    spotify_client_secret : Optional[str]
        Spotify API client secret.
    enable_beatport : bool
        Enable Beatport lookups.
    enable_tunebat : bool
        Enable TuneBat lookups.
    cache_ttl : int
        Cache time-to-live in seconds.
    """
    spotify_client_id: Optional[str] = Field(None, env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: Optional[str] = Field(None, env="SPOTIFY_CLIENT_SECRET")
    enable_beatport: bool = True
    enable_tunebat: bool = True
    cache_ttl: int = 3600  # 1 hour


class EDMConfig(BaseModel):
    """Main configuration for EDM library.

    Attributes
    ----------
    analysis : AnalysisConfig
        Analysis configuration.
    external_services : ExternalServicesConfig
        External services configuration.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : Optional[Path]
        Path to log file.
    """
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    class Config:
        env_prefix = "EDM_"


def load_config(config_path: Optional[Path] = None) -> EDMConfig:
    """Load configuration from file.

    Parameters
    ----------
    config_path : Path, optional
        Path to TOML configuration file. If not provided, looks for
        config in default location (~/.config/edm/config.toml).

    Returns
    -------
    EDMConfig
        Loaded and validated configuration.

    Raises
    ------
    ConfigurationError
        If configuration file is invalid.

    Examples
    --------
    >>> config = load_config()
    >>> print(f"Log level: {config.log_level}")
    Log level: INFO
    """
    if config_path is None:
        config_path = Path.home() / ".config" / "edm" / "config.toml"

    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        # TODO: Implement TOML loading with tomli
        # For now, return default config

    # Load from environment variables
    return EDMConfig()


def get_default_log_dir() -> Path:
    """Get the default log directory.

    Returns
    -------
    Path
        Path to log directory (~/.local/share/edm/logs/).
    """
    log_dir = Path.home() / ".local" / "share" / "edm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
