"""Tests for metadata extraction module."""

from pathlib import Path

import pytest

from edm.io.metadata import read_metadata, _get_artist, _get_title, _get_album, _get_bpm
from edm.exceptions import AudioFileError


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_read_metadata_nonexistent_file():
    """Test reading metadata from nonexistent file raises error."""
    with pytest.raises(AudioFileError, match="File not found"):
        read_metadata(Path("nonexistent.mp3"))


def test_read_metadata_wav_with_tags():
    """Test reading metadata from WAV file with ID3 tags."""
    audio_file = FIXTURES_DIR / "tagged_128bpm.wav"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    metadata = read_metadata(audio_file)

    assert isinstance(metadata, dict)
    assert metadata["artist"] == "Test Artist"
    assert metadata["title"] == "Test Track"
    assert metadata["album"] == "Test Album"
    assert metadata["bpm"] == 128.0
    assert metadata["format"] == "WAV"
    assert metadata["duration"] is not None
    assert metadata["sample_rate"] is not None


def test_read_metadata_flac_with_tags():
    """Test reading metadata from FLAC file with Vorbis comments."""
    audio_file = FIXTURES_DIR / "tagged_140bpm.flac"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    metadata = read_metadata(audio_file)

    assert metadata["artist"] == "FLAC Artist"
    assert metadata["title"] == "FLAC Track"
    assert metadata["album"] == "FLAC Album"
    assert metadata["bpm"] == 140.0
    assert metadata["format"] == "FLAC"


def test_read_metadata_no_bpm_tag():
    """Test reading metadata from file without BPM tag."""
    audio_file = FIXTURES_DIR / "no_bpm_tag.wav"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    metadata = read_metadata(audio_file)

    assert metadata["artist"] == "No BPM Artist"
    assert metadata["title"] == "No BPM"
    assert metadata["album"] == "Test"
    assert metadata["bpm"] is None  # No BPM tag present


def test_read_metadata_untagged_file():
    """Test reading metadata from untagged audio file."""
    audio_file = FIXTURES_DIR / "click_120bpm.wav"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    metadata = read_metadata(audio_file)

    # Should fall back to filename for title
    assert metadata["title"] == "click_120bpm"
    assert metadata["artist"] is None
    assert metadata["album"] is None
    assert metadata["bpm"] is None
    assert metadata["duration"] is not None
    assert metadata["format"] == "WAV"


def test_read_metadata_all_fields_present():
    """Test that all expected metadata fields are returned."""
    audio_file = FIXTURES_DIR / "tagged_128bpm.wav"
    metadata = read_metadata(audio_file)

    expected_fields = {"artist", "title", "album", "duration", "bitrate", "sample_rate", "format", "bpm"}
    assert set(metadata.keys()) == expected_fields


def test_bpm_validation_out_of_range():
    """Test that invalid BPM values are rejected."""
    # Create a temporary file with invalid BPM
    import mutagen
    from mutagen.id3 import TBPM
    
    audio_file = FIXTURES_DIR / "click_120bpm.wav"
    
    # Test with mutagen directly for BPM validation
    audio = mutagen.File(audio_file)
    
    # Mock invalid BPM scenarios
    from unittest.mock import Mock
    mock_audio = Mock()
    mock_audio.tags = {'TBPM': Mock()}
    mock_audio.tags['TBPM'].__str__ = Mock(return_value='500')  # Too high
    
    result = _get_bpm(mock_audio, audio_file)
    assert result is None  # Should reject invalid BPM


def test_title_fallback_to_filename():
    """Test that title falls back to filename when no tag present."""
    audio_file = FIXTURES_DIR / "beat_150bpm.wav"
    
    import mutagen
    audio = mutagen.File(audio_file)
    
    title = _get_title(audio, audio_file)
    assert title == "beat_150bpm"


def test_get_artist_returns_none_for_no_tags():
    """Test that _get_artist returns None when no tags present."""
    audio_file = FIXTURES_DIR / "click_125bpm.wav"
    
    import mutagen
    audio = mutagen.File(audio_file)
    
    artist = _get_artist(audio)
    assert artist is None


def test_get_album_returns_none_for_no_tags():
    """Test that _get_album returns None when no tags present."""
    audio_file = FIXTURES_DIR / "click_125bpm.wav"
    
    import mutagen
    audio = mutagen.File(audio_file)
    
    album = _get_album(audio)
    assert album is None


def test_metadata_extraction_multiple_formats():
    """Test metadata extraction works across different audio formats."""
    test_files = [
        ("tagged_128bpm.wav", "WAV", 128.0),
        ("tagged_140bpm.flac", "FLAC", 140.0),
    ]
    
    for filename, expected_format, expected_bpm in test_files:
        audio_file = FIXTURES_DIR / filename
        if audio_file.exists():
            metadata = read_metadata(audio_file)
            assert metadata["format"] == expected_format
            assert metadata["bpm"] == expected_bpm
            assert metadata["duration"] is not None
