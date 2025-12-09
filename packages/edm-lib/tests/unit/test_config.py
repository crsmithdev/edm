"""Tests for configuration module."""

from edm.config import AnalysisConfig, EDMConfig, load_config


def test_analysis_config_defaults():
    """Test AnalysisConfig default values."""
    config = AnalysisConfig()
    assert config.detect_bpm is True
    assert config.detect_structure is True
    assert config.use_madmom is True
    assert config.use_librosa is False


def test_edm_config_defaults():
    """Test EDMConfig default values."""
    config = EDMConfig()
    assert config.log_level == "INFO"
    assert config.analysis.detect_bpm is True


def test_load_config_no_file():
    """Test loading config when no file exists."""
    config = load_config()
    assert isinstance(config, EDMConfig)
    assert config.log_level == "INFO"
