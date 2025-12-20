"""Comprehensive tests for waveform generation API.

Tests GET /api/load/<filename> endpoint with:
- Waveform generation
- Caching behavior (if implemented)
- Multi-channel audio processing
- Corrupted file handling
- Edge cases (empty files, very long files)
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml


def reinitialize_services(app, temp_audio_dir, temp_annotation_dir):
    """Helper to reinitialize services with test directories."""
    from edm_annotator.services import AnnotationService, AudioService, WaveformService

    app.config["AUDIO_DIR"] = temp_audio_dir
    app.config["ANNOTATION_DIR"] = temp_annotation_dir

    app.audio_service = AudioService(config=app.config)
    app.annotation_service = AnnotationService(config=app.config, audio_service=app.audio_service)
    app.waveform_service = WaveformService(config=app.config, audio_service=app.audio_service)


def create_audio_file(path: Path, duration: float = 1.0, channels: int = 1):
    """Create a simple audio file for testing."""
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    if channels == 2:
        # Create stereo by duplicating mono
        audio_data = np.column_stack([audio_data, audio_data])

    sf.write(path, audio_data, sample_rate)


class TestLoadTrackWaveformAPI:
    """Tests for GET /api/load/<filename> endpoint."""

    def test_load_track_valid_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Loads track and generates waveform data successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test_track.wav"
        create_audio_file(audio_file)

        response = client.get("/api/load/test_track.wav")

        assert response.status_code == 200
        data = response.json

        # Verify all required fields are present
        assert "filename" in data
        assert "duration" in data
        assert "bpm" in data
        assert "downbeat" in data
        assert "sample_rate" in data
        assert "waveform_bass" in data
        assert "waveform_mids" in data
        assert "waveform_highs" in data
        assert "waveform_times" in data

    def test_load_track_waveform_structure(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Waveform data has correct structure."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        # All waveform arrays should be lists
        assert isinstance(data["waveform_bass"], list)
        assert isinstance(data["waveform_mids"], list)
        assert isinstance(data["waveform_highs"], list)
        assert isinstance(data["waveform_times"], list)

        # All should have same length
        length = len(data["waveform_bass"])
        assert len(data["waveform_mids"]) == length
        assert len(data["waveform_highs"]) == length
        assert len(data["waveform_times"]) == length

        # Should have data points
        assert length > 0

    def test_load_track_metadata(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns correct metadata."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=2.0)

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        assert data["filename"] == "test.wav"
        assert isinstance(data["duration"], float)
        assert 1.9 <= data["duration"] <= 2.1  # ~2 seconds
        assert data["sample_rate"] == 22050

    def test_load_track_with_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Loads BPM and boundaries from existing annotation."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "annotated.wav"
        create_audio_file(audio_file)

        # Create reference annotation
        ref_dir = temp_annotation_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        annotation = {
            "audio": {
                "bpm": 128.5,
                "downbeat": 0.25,
                "duration": 180.0,
            },
            "structure": [
                {"time": 0.0, "label": "intro"},
                {"time": 30.0, "label": "buildup"},
            ],
        }
        (ref_dir / "annotated.yaml").write_text(yaml.dump(annotation))

        response = client.get("/api/load/annotated.wav")

        assert response.status_code == 200
        data = response.json

        assert data["bpm"] == 128.5
        assert data["downbeat"] == 0.25
        assert data["annotation_tier"] == 1  # Reference tier
        assert len(data["boundaries"]) == 2

    def test_load_track_no_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns null BPM when no annotation exists."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "unannotated.wav"
        create_audio_file(audio_file)

        response = client.get("/api/load/unannotated.wav")

        assert response.status_code == 200
        data = response.json

        assert data["bpm"] is None
        assert data["downbeat"] == 0.0
        assert data["boundaries"] is None
        assert data["annotation_tier"] is None

    def test_load_track_not_found(self, client, temp_audio_dir, app):
        """Returns 404 for non-existent files."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/load/nonexistent.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_load_track_path_traversal(self, client, temp_audio_dir, app):
        """Returns 400 for path traversal attempts."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/load/../../../etc/passwd")

        assert response.status_code == 400
        assert "error" in response.json

    def test_load_track_corrupted_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles corrupted audio files gracefully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create corrupted file
        corrupted_file = temp_audio_dir / "corrupted.wav"
        corrupted_file.write_bytes(b"INVALID AUDIO DATA")

        response = client.get("/api/load/corrupted.wav")

        # Should return error
        assert response.status_code in [400, 500]
        assert "error" in response.json

    def test_load_track_stereo_audio(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles stereo audio files correctly."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "stereo.wav"
        create_audio_file(audio_file, duration=2.0, channels=2)

        response = client.get("/api/load/stereo.wav")

        assert response.status_code == 200
        data = response.json

        # Should generate waveform (mono mix or first channel)
        assert len(data["waveform_bass"]) > 0
        assert len(data["waveform_mids"]) > 0
        assert len(data["waveform_highs"]) > 0

    def test_load_track_long_audio(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles long audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create 30-second file
        audio_file = temp_audio_dir / "long.wav"
        create_audio_file(audio_file, duration=30.0)

        response = client.get("/api/load/long.wav")

        assert response.status_code == 200
        data = response.json

        assert 29.0 <= data["duration"] <= 31.0
        # Should have many waveform points
        assert len(data["waveform_bass"]) > 100

    def test_load_track_short_audio(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles very short audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create 0.1-second file
        audio_file = temp_audio_dir / "short.wav"
        create_audio_file(audio_file, duration=0.1)

        response = client.get("/api/load/short.wav")

        assert response.status_code == 200
        data = response.json

        assert data["duration"] < 0.2
        # Should still generate some waveform points
        assert len(data["waveform_bass"]) > 0

    def test_load_track_silent_audio(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles silent audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create silent audio
        sample_rate = 22050
        duration = 1.0
        silent = np.zeros(int(sample_rate * duration), dtype=np.float32)

        audio_file = temp_audio_dir / "silent.wav"
        sf.write(audio_file, silent, sample_rate)

        response = client.get("/api/load/silent.wav")

        assert response.status_code == 200
        data = response.json

        # All RMS values should be near zero
        assert all(x < 0.01 for x in data["waveform_bass"])
        assert all(x < 0.01 for x in data["waveform_mids"])
        assert all(x < 0.01 for x in data["waveform_highs"])

    def test_load_track_multiband_signal(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Verifies 3-band separation."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create signal with bass, mids, and highs
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        bass = 0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz
        mids = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz
        highs = 0.2 * np.sin(2 * np.pi * 8000 * t)  # 8000 Hz
        signal = (bass + mids + highs).astype(np.float32)

        audio_file = temp_audio_dir / "multiband.wav"
        sf.write(audio_file, signal, sample_rate)

        response = client.get("/api/load/multiband.wav")

        assert response.status_code == 200
        data = response.json

        # Each band should have energy
        bass_energy = np.mean(data["waveform_bass"])
        mids_energy = np.mean(data["waveform_mids"])
        highs_energy = np.mean(data["waveform_highs"])

        assert bass_energy > 0.01
        assert mids_energy > 0.01
        assert highs_energy > 0.01

    def test_load_track_waveform_generation_error(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles waveform generation errors gracefully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        # Mock generate_waveform to raise exception
        with patch.object(
            app.waveform_service, "generate_waveform", side_effect=RuntimeError("DSP error")
        ):
            response = client.get("/api/load/test.wav")

            assert response.status_code == 500
            assert "error" in response.json

    def test_load_track_annotation_tier_priority(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Prefers reference annotations over generated."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        # Create both reference and generated annotations
        ref_dir = temp_annotation_dir / "reference"
        gen_dir = temp_annotation_dir / "generated"
        ref_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_annotation = {"audio": {"bpm": 128.0, "downbeat": 0.0}}
        gen_annotation = {"audio": {"bpm": 140.0, "downbeat": 0.5}}

        (ref_dir / "test.yaml").write_text(yaml.dump(ref_annotation))
        (gen_dir / "test.yaml").write_text(yaml.dump(gen_annotation))

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        # Should use reference annotation (tier 1)
        assert data["bpm"] == 128.0
        assert data["downbeat"] == 0.0
        assert data["annotation_tier"] == 1

    def test_load_track_times_monotonic(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Waveform times are monotonically increasing."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=5.0)

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        times = data["waveform_times"]
        # Verify monotonically increasing
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_load_track_times_coverage(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Waveform times cover entire duration."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=3.0)

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        times = data["waveform_times"]
        duration = data["duration"]

        # First time should be near 0
        assert times[0] < 0.1

        # Last time should be near duration
        assert abs(times[-1] - duration) < 0.1

    def test_load_track_rms_non_negative(self, client, temp_audio_dir, temp_annotation_dir, app):
        """All RMS values are non-negative."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        response = client.get("/api/load/test.wav")

        assert response.status_code == 200
        data = response.json

        assert all(x >= 0 for x in data["waveform_bass"])
        assert all(x >= 0 for x in data["waveform_mids"])
        assert all(x >= 0 for x in data["waveform_highs"])

    def test_load_track_special_characters_filename(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles special characters in filenames."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "Artist - Track (Remix).wav"
        create_audio_file(audio_file)

        response = client.get("/api/load/Artist - Track (Remix).wav")

        assert response.status_code == 200
        data = response.json
        assert data["filename"] == "Artist - Track (Remix).wav"

    def test_load_track_caching_behavior(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Tests caching behavior (if implemented)."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        # First request
        response1 = client.get("/api/load/test.wav")
        assert response1.status_code == 200
        data1 = response1.json

        # Second request (might be cached)
        response2 = client.get("/api/load/test.wav")
        assert response2.status_code == 200
        data2 = response2.json

        # Data should be identical
        assert data1["waveform_bass"] == data2["waveform_bass"]
        assert data1["waveform_mids"] == data2["waveform_mids"]
        assert data1["waveform_highs"] == data2["waveform_highs"]

    def test_load_track_empty_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles empty audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create empty file
        audio_file = temp_audio_dir / "empty.wav"
        audio_file.write_bytes(b"")

        response = client.get("/api/load/empty.wav")

        # Should return error
        assert response.status_code in [400, 500]
        assert "error" in response.json
