"""Tests for configuration management."""

from edm_annotator.config import DevelopmentConfig, ProductionConfig, TestingConfig


def test_development_config():
    """Test development configuration."""
    assert DevelopmentConfig.DEBUG is True
    assert DevelopmentConfig.TESTING is False
    assert DevelopmentConfig.WAVEFORM_SAMPLE_RATE == 22050


def test_production_config():
    """Test production configuration."""
    assert ProductionConfig.DEBUG is False
    assert ProductionConfig.TESTING is False
    assert ProductionConfig.CORS_ORIGINS == []


def test_testing_config():
    """Test testing configuration."""
    assert TestingConfig.DEBUG is False
    assert TestingConfig.TESTING is True
    assert "fixtures" in str(TestingConfig.AUDIO_DIR)


def test_valid_labels():
    """Test valid section labels are defined."""
    assert "intro" in DevelopmentConfig.VALID_LABELS
    assert "buildup" in DevelopmentConfig.VALID_LABELS
    assert "breakdown" in DevelopmentConfig.VALID_LABELS
    assert "breakbuild" in DevelopmentConfig.VALID_LABELS
    assert "outro" in DevelopmentConfig.VALID_LABELS
    assert "unlabeled" in DevelopmentConfig.VALID_LABELS
