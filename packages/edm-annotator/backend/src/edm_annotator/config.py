"""Configuration management for EDM Annotator.

Provides environment-specific configurations with sensible defaults.
"""

import os
from pathlib import Path


class Config:
    """Base configuration - shared settings."""

    # Determine directory paths
    PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
    MONOREPO_ROOT = PACKAGE_ROOT.parent.parent
    TEMPLATE_DIR = PACKAGE_ROOT / "templates"
    STATIC_DIR = PACKAGE_ROOT / "static"

    # Audio and annotation directories (configurable via environment)
    AUDIO_DIR = Path(os.getenv("EDM_AUDIO_DIR", Path.home() / "music"))
    ANNOTATION_DIR = Path(os.getenv("EDM_ANNOTATION_DIR", MONOREPO_ROOT / "data" / "annotations"))

    # Waveform processing parameters
    WAVEFORM_SAMPLE_RATE = 22050
    WAVEFORM_HOP_LENGTH = 128  # ~5.8ms at 22050 Hz
    WAVEFORM_FRAME_LENGTH = 1024

    # Frequency band ranges (Hz)
    BASS_LOW = 20
    BASS_HIGH = 250
    MIDS_LOW = 250
    MIDS_HIGH = 4000
    HIGHS_LOW = 4000

    # Valid section labels
    VALID_LABELS = ["intro", "buildup", "breakdown", "breakbuild", "outro", "unlabeled"]

    # Supported audio formats
    AUDIO_EXTENSIONS = ["*.mp3", "*.flac", "*.wav", "*.m4a"]

    # CORS settings
    CORS_ORIGINS = ["http://localhost:5174"]  # Vite dev server

    # Flask settings
    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    """Development environment configuration."""

    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production environment configuration."""

    DEBUG = False
    TESTING = False
    CORS_ORIGINS = []  # Served from same origin in production


class TestingConfig(Config):
    """Testing environment configuration."""

    DEBUG = False
    TESTING = True
    AUDIO_DIR = Path("tests/fixtures/audio")
    ANNOTATION_DIR = Path("tests/fixtures/annotations")


# Configuration class mapping
config_class_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
