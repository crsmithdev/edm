"""Pytest configuration and fixtures for EDM Annotator backend tests."""

import numpy as np
import pytest
import soundfile as sf

from edm_annotator.app import create_app


@pytest.fixture
def app():
    """Create Flask app configured for testing."""
    test_app = create_app("testing")
    yield test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create temporary audio directory with sample files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_annotation_dir(tmp_path):
    """Create temporary annotation directory structure."""
    annotation_dir = tmp_path / "annotations"
    reference_dir = annotation_dir / "reference"
    generated_dir = annotation_dir / "generated"
    reference_dir.mkdir(parents=True)
    generated_dir.mkdir(parents=True)
    return annotation_dir


@pytest.fixture
def sample_audio_file(temp_audio_dir):
    """Generate a 1-second test audio file.

    Returns:
        Path to generated audio file
    """
    # Generate 1 second of 440 Hz sine wave
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Save as WAV file
    audio_path = temp_audio_dir / "test_track.wav"
    sf.write(audio_path, audio_data, sample_rate)

    return audio_path
