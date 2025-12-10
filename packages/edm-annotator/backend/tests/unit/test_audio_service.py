"""Tests for AudioService class."""

from unittest.mock import patch

import numpy as np
import pytest

from edm_annotator.config import TestingConfig
from edm_annotator.services.audio_service import AudioService


@pytest.fixture
def audio_config(temp_audio_dir):
    """Create test configuration for AudioService."""
    config = {
        "AUDIO_DIR": temp_audio_dir,
        "WAVEFORM_SAMPLE_RATE": TestingConfig.WAVEFORM_SAMPLE_RATE,
        "AUDIO_EXTENSIONS": TestingConfig.AUDIO_EXTENSIONS,
    }
    return config


@pytest.fixture
def audio_service(audio_config):
    """Create AudioService instance for testing."""
    return AudioService(audio_config)


class TestValidateAudioPath:
    """Tests for validate_audio_path method."""

    def test_validate_valid_file(self, audio_service, sample_audio_file):
        """Test validation of existing audio file."""
        result = audio_service.validate_audio_path(sample_audio_file.name)
        assert result == sample_audio_file
        assert result.exists()

    def test_validate_missing_file(self, audio_service):
        """Test validation raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            audio_service.validate_audio_path("nonexistent.wav")

    def test_validate_blocks_parent_traversal(self, audio_service):
        """Test validation blocks path traversal with parent directory."""
        with pytest.raises(ValueError, match="Invalid filename"):
            audio_service.validate_audio_path("../etc/passwd")

    def test_validate_blocks_double_dots(self, audio_service):
        """Test validation blocks any path with double dots."""
        with pytest.raises(ValueError, match="Invalid filename"):
            audio_service.validate_audio_path("subdir/../track.wav")

    def test_validate_blocks_absolute_path(self, audio_service):
        """Test validation blocks absolute paths."""
        with pytest.raises(ValueError, match="Invalid filename"):
            audio_service.validate_audio_path("/absolute/path/track.wav")

    def test_validate_accepts_simple_filename(self, audio_service, temp_audio_dir):
        """Test validation accepts simple filename without path components."""
        # Create test file
        test_file = temp_audio_dir / "valid_track.wav"
        test_file.touch()

        result = audio_service.validate_audio_path("valid_track.wav")
        assert result == test_file


class TestLoadAudio:
    """Tests for load_audio method."""

    def test_load_valid_audio(self, audio_service, sample_audio_file):
        """Test loading valid audio file."""
        audio_data, sample_rate = audio_service.load_audio(sample_audio_file.name)

        assert isinstance(audio_data, np.ndarray)
        assert sample_rate == TestingConfig.WAVEFORM_SAMPLE_RATE
        assert len(audio_data) > 0

    def test_load_missing_file_raises_error(self, audio_service):
        """Test loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            audio_service.load_audio("missing.wav")

    def test_load_resamples_to_target_rate(self, audio_service, temp_audio_dir):
        """Test audio is resampled to configured sample rate."""
        import soundfile as sf

        # Create audio at different sample rate
        original_sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(original_sr * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_path = temp_audio_dir / "44khz_track.wav"
        sf.write(audio_path, audio_data, original_sr)

        # Load and verify resampling
        loaded_audio, loaded_sr = audio_service.load_audio("44khz_track.wav")
        assert loaded_sr == TestingConfig.WAVEFORM_SAMPLE_RATE

    @patch("edm_annotator.services.audio_service.load_audio")
    def test_load_handles_corrupt_file(self, mock_load_audio, audio_service, temp_audio_dir):
        """Test loading corrupt file raises appropriate error."""
        # Create corrupt file
        corrupt_file = temp_audio_dir / "corrupt.wav"
        corrupt_file.write_bytes(b"not a valid wav file")

        # Mock to raise exception
        mock_load_audio.side_effect = Exception("Failed to load audio")

        with pytest.raises(Exception, match="Failed to load audio"):
            audio_service.load_audio("corrupt.wav")


class TestGetDuration:
    """Tests for get_duration method."""

    def test_get_duration_accurate(self, audio_service, sample_audio_file):
        """Test duration calculation is accurate."""
        duration = audio_service.get_duration(sample_audio_file.name)

        # Sample audio is 1 second, allow small tolerance
        assert 0.95 <= duration <= 1.05

    def test_get_duration_missing_file(self, audio_service):
        """Test get_duration raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            audio_service.get_duration("missing.wav")

    def test_get_duration_multiple_lengths(self, audio_service, temp_audio_dir):
        """Test duration calculation for different file lengths."""
        import soundfile as sf

        durations = [0.5, 2.0, 5.0]

        for expected_duration in durations:
            # Generate audio of specific duration
            sr = 22050
            t = np.linspace(0, expected_duration, int(sr * expected_duration))
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            filename = f"track_{expected_duration}s.wav"
            audio_path = temp_audio_dir / filename
            sf.write(audio_path, audio_data, sr)

            # Verify duration
            actual_duration = audio_service.get_duration(filename)
            assert abs(actual_duration - expected_duration) < 0.01


class TestListAudioFiles:
    """Tests for list_audio_files method."""

    def test_list_empty_directory(self, audio_service):
        """Test listing empty audio directory."""
        files = audio_service.list_audio_files()
        assert files == []

    def test_list_single_file(self, audio_service, sample_audio_file):
        """Test listing directory with single audio file."""
        files = audio_service.list_audio_files()
        assert len(files) == 1
        assert files[0] == sample_audio_file

    def test_list_multiple_files(self, audio_service, temp_audio_dir):
        """Test listing directory with multiple audio files."""
        # Create multiple files
        filenames = ["track1.wav", "track2.mp3", "track3.flac"]
        for filename in filenames:
            (temp_audio_dir / filename).touch()

        files = audio_service.list_audio_files()
        assert len(files) == 3

        # Verify sorted by name
        file_names = [f.name for f in files]
        assert file_names == sorted(filenames)

    def test_list_filters_by_extension(self, audio_service, temp_audio_dir):
        """Test listing only includes configured audio extensions."""
        # Create audio and non-audio files
        (temp_audio_dir / "track.wav").touch()
        (temp_audio_dir / "track.mp3").touch()
        (temp_audio_dir / "document.txt").touch()
        (temp_audio_dir / "image.jpg").touch()

        files = audio_service.list_audio_files()
        file_names = [f.name for f in files]

        assert "track.wav" in file_names
        assert "track.mp3" in file_names
        assert "document.txt" not in file_names
        assert "image.jpg" not in file_names

    def test_list_sorted_alphabetically(self, audio_service, temp_audio_dir):
        """Test files are sorted alphabetically by name."""
        # Create files in non-alphabetical order
        filenames = ["zebra.wav", "apple.mp3", "banana.flac", "cherry.wav"]
        for filename in filenames:
            (temp_audio_dir / filename).touch()

        files = audio_service.list_audio_files()
        file_names = [f.name for f in files]

        assert file_names == sorted(filenames)

    def test_list_all_configured_extensions(self, audio_service, temp_audio_dir):
        """Test listing includes all configured audio extensions."""
        # Create one file of each extension
        extensions = ["mp3", "flac", "wav", "m4a"]
        for ext in extensions:
            (temp_audio_dir / f"track.{ext}").touch()

        files = audio_service.list_audio_files()
        assert len(files) == 4

    def test_list_ignores_subdirectories(self, audio_service, temp_audio_dir):
        """Test listing only includes files in top-level directory."""
        # Create files in subdirectory
        subdir = temp_audio_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.wav").touch()
        (temp_audio_dir / "top_level.wav").touch()

        files = audio_service.list_audio_files()

        # Should only find top-level file
        assert len(files) == 1
        assert files[0].name == "top_level.wav"
