"""Comprehensive tests for tracks API endpoint.

Tests GET /api/tracks endpoint with:
- Empty directory handling
- Permission errors
- Non-audio file filtering
- Annotation status accuracy
- Subdirectory handling
- Mixed format support
- Sorting validation
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def reinitialize_services(app, temp_audio_dir, temp_annotation_dir):
    """Helper to reinitialize services with test directories."""
    from edm_annotator.services import AnnotationService, AudioService, WaveformService

    app.config["AUDIO_DIR"] = temp_audio_dir
    app.config["ANNOTATION_DIR"] = temp_annotation_dir

    app.audio_service = AudioService(config=app.config)
    app.annotation_service = AnnotationService(config=app.config, audio_service=app.audio_service)
    app.waveform_service = WaveformService(config=app.config, audio_service=app.audio_service)


def create_audio_file(path: Path, duration: float = 1.0):
    """Create a simple audio file for testing."""
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, audio_data, sample_rate)


class TestListTracksAPI:
    """Tests for GET /api/tracks endpoint."""

    def test_list_tracks_empty_directory(self, client, temp_audio_dir, app):
        """Returns empty array when no audio files exist."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/tracks")

        assert response.status_code == 200
        assert response.json == []

    def test_list_tracks_with_files(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns valid JSON array with track metadata."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create sample audio files
        create_audio_file(temp_audio_dir / "track1.wav")
        create_audio_file(temp_audio_dir / "track2.mp3")
        create_audio_file(temp_audio_dir / "track3.flac")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 3

        # Verify required fields
        for track in tracks:
            assert "filename" in track
            assert "path" in track
            assert "has_reference" in track
            assert "has_generated" in track

    def test_list_tracks_alphabetical_sorting(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Returns tracks sorted alphabetically by filename."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create files in non-alphabetical order
        create_audio_file(temp_audio_dir / "zebra.wav")
        create_audio_file(temp_audio_dir / "alpha.wav")
        create_audio_file(temp_audio_dir / "beta.wav")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        filenames = [track["filename"] for track in tracks]
        assert filenames == ["alpha.wav", "beta.wav", "zebra.wav"]

    def test_list_tracks_annotation_status_reference(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Correctly identifies reference annotations."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "track1.wav")

        # Create reference annotation
        ref_dir = temp_annotation_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "track1.yaml").write_text("audio:\n  bpm: 128\n")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1
        assert tracks[0]["has_reference"] is True
        assert tracks[0]["has_generated"] is False

    def test_list_tracks_annotation_status_generated(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Correctly identifies generated annotations."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "track1.wav")

        # Create generated annotation
        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "track1.yaml").write_text("audio:\n  bpm: 140\n")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1
        assert tracks[0]["has_reference"] is False
        assert tracks[0]["has_generated"] is True

    def test_list_tracks_annotation_status_both(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Correctly identifies both reference and generated annotations."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "track1.wav")

        # Create both types of annotations
        ref_dir = temp_annotation_dir / "reference"
        gen_dir = temp_annotation_dir / "generated"
        ref_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "track1.yaml").write_text("audio:\n  bpm: 128\n")
        (gen_dir / "track1.yaml").write_text("audio:\n  bpm: 140\n")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1
        assert tracks[0]["has_reference"] is True
        assert tracks[0]["has_generated"] is True

    def test_list_tracks_annotation_status_none(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Correctly identifies tracks with no annotations."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "track1.wav")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1
        assert tracks[0]["has_reference"] is False
        assert tracks[0]["has_generated"] is False

    def test_list_tracks_filters_non_audio_files(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Filters out non-audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio files
        create_audio_file(temp_audio_dir / "track1.wav")
        create_audio_file(temp_audio_dir / "track2.mp3")

        # Create non-audio files
        (temp_audio_dir / "readme.txt").write_text("Not audio")
        (temp_audio_dir / "data.json").write_text("{}")
        (temp_audio_dir / "cover.jpg").write_bytes(b"fake image data")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        # Should only include audio files
        assert len(tracks) == 2
        filenames = [track["filename"] for track in tracks]
        assert "track1.wav" in filenames
        assert "track2.mp3" in filenames
        assert "readme.txt" not in filenames

    def test_list_tracks_mixed_audio_formats(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Correctly handles mixed audio formats."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create files in different formats (skip m4a as soundfile doesn't support it)
        create_audio_file(temp_audio_dir / "track1.wav")
        create_audio_file(temp_audio_dir / "track2.mp3")
        create_audio_file(temp_audio_dir / "track3.flac")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        assert len(tracks) == 3
        extensions = [Path(track["filename"]).suffix for track in tracks]
        assert ".wav" in extensions
        assert ".mp3" in extensions
        assert ".flac" in extensions

    def test_list_tracks_subdirectory_handling(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles subdirectories according to configuration."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create subdirectory with audio files
        subdir = temp_audio_dir / "artist"
        subdir.mkdir()
        create_audio_file(temp_audio_dir / "root_track.wav")
        create_audio_file(subdir / "sub_track.wav")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        # Current implementation doesn't recurse into subdirectories
        # Only root_track.wav should be found
        assert len(tracks) == 1
        assert tracks[0]["filename"] == "root_track.wav"

    def test_list_tracks_path_calculation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Path field contains valid path string."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "test.wav")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1

        # Path should be a string containing the filename
        path = tracks[0]["path"]
        assert isinstance(path, str)
        assert "test.wav" in path

    def test_list_tracks_permission_error(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Tests permission error handling on directory."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create a file
        create_audio_file(temp_audio_dir / "test.wav")

        # Try to make directory unreadable (Unix only)
        try:
            import os

            original_perms = os.stat(temp_audio_dir).st_mode
            os.chmod(temp_audio_dir, 0o000)

            response = client.get("/api/tracks")

            # Restore permissions
            os.chmod(temp_audio_dir, original_perms)

            # Should return error or empty list
            assert response.status_code in [200, 500]
        except (OSError, PermissionError):
            # Skip test if we can't change permissions
            pytest.skip("Cannot test permission errors on this system")

    def test_list_tracks_case_sensitivity(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles case-sensitive filenames correctly."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create files with different cases
        create_audio_file(temp_audio_dir / "Track.wav")
        create_audio_file(temp_audio_dir / "track.mp3")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        # Both files should be listed
        assert len(tracks) == 2

    def test_list_tracks_special_characters_in_filename(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles special characters in filenames."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create files with special characters
        create_audio_file(temp_audio_dir / "Artist - Track (Remix).wav")
        create_audio_file(temp_audio_dir / "Track #1.mp3")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        assert len(tracks) == 2
        filenames = [track["filename"] for track in tracks]
        assert "Artist - Track (Remix).wav" in filenames
        assert "Track #1.mp3" in filenames

    def test_list_tracks_unicode_filenames(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles Unicode characters in filenames."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create files with Unicode characters
        create_audio_file(temp_audio_dir / "Künstler.wav")
        create_audio_file(temp_audio_dir / "日本語.mp3")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        assert len(tracks) == 2

    def test_list_tracks_very_long_filename(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles very long filenames."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create file with long name (but within filesystem limits)
        long_name = "a" * 200 + ".wav"
        create_audio_file(temp_audio_dir / long_name)

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json

        assert len(tracks) == 1
        assert tracks[0]["filename"] == long_name

    def test_list_tracks_multiple_tracks_annotation_accuracy(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Annotation status is accurate for multiple tracks."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create multiple tracks
        create_audio_file(temp_audio_dir / "track1.wav")
        create_audio_file(temp_audio_dir / "track2.wav")
        create_audio_file(temp_audio_dir / "track3.wav")

        # Create annotations for some tracks
        ref_dir = temp_annotation_dir / "reference"
        gen_dir = temp_annotation_dir / "generated"
        ref_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

        (ref_dir / "track1.yaml").write_text("audio:\n  bpm: 128\n")
        (gen_dir / "track2.yaml").write_text("audio:\n  bpm: 140\n")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 3

        # Find each track and verify annotation status
        track1 = next(t for t in tracks if t["filename"] == "track1.wav")
        track2 = next(t for t in tracks if t["filename"] == "track2.wav")
        track3 = next(t for t in tracks if t["filename"] == "track3.wav")

        assert track1["has_reference"] is True
        assert track1["has_generated"] is False

        assert track2["has_reference"] is False
        assert track2["has_generated"] is True

        assert track3["has_reference"] is False
        assert track3["has_generated"] is False

    def test_list_tracks_empty_annotation_directory(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles missing annotation directories gracefully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        create_audio_file(temp_audio_dir / "track1.wav")

        # Remove annotation directories
        import shutil

        if (temp_annotation_dir / "reference").exists():
            shutil.rmtree(temp_annotation_dir / "reference")
        if (temp_annotation_dir / "generated").exists():
            shutil.rmtree(temp_annotation_dir / "generated")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1
        assert tracks[0]["has_reference"] is False
        assert tracks[0]["has_generated"] is False
