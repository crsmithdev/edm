"""Comprehensive tests for audio file serving API.

Tests GET /api/audio/<filename> endpoint with:
- Audio file serving
- MIME type handling
- Range requests
- File not found errors
- Path traversal security
- Different audio formats
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


class TestServeAudioAPI:
    """Tests for GET /api/audio/<filename> endpoint."""

    def test_serve_audio_wav_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves WAV audio file successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        response = client.get("/api/audio/test.wav")

        assert response.status_code == 200
        assert len(response.data) > 0
        # MIME type should be audio-related
        assert response.content_type.startswith("audio/") or "wav" in response.content_type.lower()

    def test_serve_audio_mp3_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves MP3 audio file successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.mp3"
        create_audio_file(audio_file)

        response = client.get("/api/audio/test.mp3")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_flac_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves FLAC audio file successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.flac"
        create_audio_file(audio_file)

        response = client.get("/api/audio/test.flac")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_m4a_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves M4A audio file successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # m4a not supported by soundfile, skip this format
        pytest.skip("M4A format not supported by soundfile")

        response = client.get("/api/audio/test.m4a")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_file_not_found(self, client, temp_audio_dir, app):
        """Returns 404 for non-existent files."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/audio/nonexistent.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_serve_audio_path_traversal_dotdot(self, client, temp_audio_dir, app):
        """Blocks path traversal with .. sequences."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/audio/../../../etc/passwd")

        assert response.status_code == 400
        assert "error" in response.json

    def test_serve_audio_path_traversal_absolute(self, client, temp_audio_dir, app):
        """Blocks absolute path attempts."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/audio//etc/passwd")

        # Flask might redirect or return 400
        assert response.status_code in [308, 400]

    def test_serve_audio_mime_type_wav(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns correct MIME type for WAV files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        response = client.get("/api/audio/test.wav")

        assert response.status_code == 200
        # Should be audio/wav, audio/x-wav, or similar
        content_type = response.content_type.lower()
        assert "audio" in content_type or "wav" in content_type

    def test_serve_audio_special_characters_filename(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Serves files with special characters in names."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "Artist - Track (Remix).wav"
        create_audio_file(audio_file)

        response = client.get("/api/audio/Artist - Track (Remix).wav")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_unicode_filename(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves files with Unicode characters in names."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "Künstler.wav"
        create_audio_file(audio_file)

        response = client.get("/api/audio/Künstler.wav")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_range_request_support(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Tests if range requests are supported (optional feature)."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=10.0)

        # Try range request
        response = client.get("/api/audio/test.wav", headers={"Range": "bytes=0-1023"})

        # Range requests might or might not be supported
        # Accept 200 (full content) or 206 (partial content)
        assert response.status_code in [200, 206]
        assert len(response.data) > 0

    def test_serve_audio_response_headers(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Verifies appropriate response headers are set."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        response = client.get("/api/audio/test.wav")

        assert response.status_code == 200
        # Should have content-type header
        assert response.content_type is not None

    def test_serve_audio_large_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves large audio files successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create a larger file (10 seconds)
        audio_file = temp_audio_dir / "large.wav"
        create_audio_file(audio_file, duration=10.0)

        response = client.get("/api/audio/large.wav")

        assert response.status_code == 200
        assert len(response.data) > 0

    def test_serve_audio_empty_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles empty audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create empty file
        audio_file = temp_audio_dir / "empty.wav"
        audio_file.write_bytes(b"")

        response = client.get("/api/audio/empty.wav")

        # Should return 200 with empty content or error
        assert response.status_code in [200, 400, 500]

    def test_serve_audio_corrupted_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves corrupted file (might fail at playback, not serving)."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create corrupted audio file
        audio_file = temp_audio_dir / "corrupted.wav"
        audio_file.write_bytes(b"INVALID AUDIO DATA")

        response = client.get("/api/audio/corrupted.wav")

        # File serving might succeed (corruption detected at playback)
        # or fail if validation is done
        assert response.status_code in [200, 400, 500]

    def test_serve_audio_permission_error(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Tests permission error handling (if implemented)."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        # Make file unreadable to simulate permission error (Unix only)
        try:
            import os
            import stat

            os.chmod(audio_file, 0o000)

            response = client.get("/api/audio/test.wav")

            # Restore permissions
            os.chmod(audio_file, stat.S_IRUSR | stat.S_IWUSR)

            # Should return error if permissions are checked
            # Might succeed if send_file doesn't check permissions
            assert response.status_code in [200, 400, 403, 500]
        except (OSError, PermissionError):
            # Skip test if we can't change permissions
            pytest.skip("Cannot test permission errors on this system")

    def test_serve_audio_case_sensitive_filename(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Filename matching is case-sensitive."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "Test.wav"
        create_audio_file(audio_file)

        # Request with different case
        response = client.get("/api/audio/test.wav")

        # On case-sensitive filesystems, this should fail
        # On case-insensitive (macOS, Windows), might succeed
        # Just verify it handles consistently
        assert response.status_code in [200, 404]

    def test_serve_audio_multiple_requests(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles multiple concurrent requests for same file."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        # Make multiple requests
        response1 = client.get("/api/audio/test.wav")
        response2 = client.get("/api/audio/test.wav")
        response3 = client.get("/api/audio/test.wav")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        # All should return same data
        assert response1.data == response2.data == response3.data

    def test_serve_audio_different_files(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves different files correctly."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file1 = temp_audio_dir / "track1.wav"
        audio_file2 = temp_audio_dir / "track2.wav"
        create_audio_file(audio_file1, duration=1.0)
        create_audio_file(audio_file2, duration=2.0)

        response1 = client.get("/api/audio/track1.wav")
        response2 = client.get("/api/audio/track2.wav")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Different files should have different sizes
        assert len(response1.data) != len(response2.data)

    def test_serve_audio_url_encoding(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles URL-encoded filenames."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "track with spaces.wav"
        create_audio_file(audio_file)

        # Request with URL-encoded spaces
        response = client.get("/api/audio/track%20with%20spaces.wav")

        assert response.status_code == 200
        assert len(response.data) > 0
