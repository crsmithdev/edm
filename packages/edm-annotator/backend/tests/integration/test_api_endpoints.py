"""Integration tests for backend API endpoints.

Tests all API endpoints with Flask test client, validating:
- HTTP status codes
- Response JSON schemas
- Error handling
- Security (path traversal)
- Data persistence
"""

import json
from pathlib import Path

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


class TestTracksEndpoint:
    """Test GET /api/tracks endpoint."""

    def test_list_tracks_empty_directory(self, client, temp_audio_dir, app):
        """Returns empty array when no audio files exist."""
        # Configure app with empty audio directory
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/tracks")

        assert response.status_code == 200
        assert response.json == []

    def test_list_tracks_with_files(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns valid JSON array with track metadata."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create sample audio files
        self._create_audio_file(temp_audio_dir / "track1.wav")
        self._create_audio_file(temp_audio_dir / "track2.mp3")
        self._create_audio_file(temp_audio_dir / "track3.flac")

        # Create reference annotation for track1
        ref_dir = temp_annotation_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "track1.yaml").write_text("audio:\n  bpm: 128\n")

        # Create generated annotation for track2
        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "track2.yaml").write_text("audio:\n  bpm: 140\n")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 3

        # Verify alphabetical sorting
        assert tracks[0]["filename"] == "track1.wav"
        assert tracks[1]["filename"] == "track2.mp3"
        assert tracks[2]["filename"] == "track3.flac"

        # Verify annotation status
        assert tracks[0]["has_reference"] is True
        assert tracks[0]["has_generated"] is False
        assert tracks[1]["has_reference"] is False
        assert tracks[1]["has_generated"] is True
        assert tracks[2]["has_reference"] is False
        assert tracks[2]["has_generated"] is False

        # Verify required fields
        for track in tracks:
            assert "filename" in track
            assert "path" in track
            assert "has_reference" in track
            assert "has_generated" in track
            # Path should be a string (may be relative or absolute in test)
            assert isinstance(track["path"], str)

    def test_list_tracks_relative_path(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Path calculation is consistent and contains track info."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        self._create_audio_file(temp_audio_dir / "test.wav")

        response = client.get("/api/tracks")

        assert response.status_code == 200
        tracks = response.json
        assert len(tracks) == 1

        # Path should be a string containing filename
        path = tracks[0]["path"]
        assert isinstance(path, str)
        assert "test.wav" in path

    @staticmethod
    def _create_audio_file(path: Path, duration: float = 1.0):
        """Create a simple audio file for testing."""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio_data, sample_rate)


class TestLoadTrackEndpoint:
    """Test GET /api/load/<filename> endpoint (CRITICAL)."""

    def test_load_track_valid_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Loads track with valid filename and generates waveform data."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test_track.wav"
        self._create_audio_file(audio_file)

        response = client.get("/api/load/test_track.wav")

        assert response.status_code == 200
        data = response.json

        # Verify response schema - all required fields present
        assert "filename" in data
        assert "duration" in data
        assert "bpm" in data
        assert "downbeat" in data
        assert "sample_rate" in data
        assert "waveform_bass" in data
        assert "waveform_mids" in data
        assert "waveform_highs" in data
        assert "waveform_times" in data

        # Verify data types and values
        assert data["filename"] == "test_track.wav"
        assert isinstance(data["duration"], float)
        assert data["duration"] > 0
        assert data["sample_rate"] == 22050

        # Verify 3-band waveform data
        assert isinstance(data["waveform_bass"], list)
        assert isinstance(data["waveform_mids"], list)
        assert isinstance(data["waveform_highs"], list)
        assert isinstance(data["waveform_times"], list)
        assert len(data["waveform_bass"]) > 0
        assert len(data["waveform_mids"]) == len(data["waveform_bass"])
        assert len(data["waveform_highs"]) == len(data["waveform_bass"])
        assert len(data["waveform_times"]) == len(data["waveform_bass"])

    def test_load_track_with_existing_annotation(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Loads BPM and downbeat from existing annotation."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "annotated_track.wav"
        self._create_audio_file(audio_file)

        # Create reference annotation
        ref_dir = temp_annotation_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        annotation = {
            "audio": {
                "bpm": 128.5,
                "downbeat": 0.25,
                "duration": 180.0,
            }
        }
        (ref_dir / "annotated_track.yaml").write_text(yaml.dump(annotation))

        response = client.get("/api/load/annotated_track.wav")

        assert response.status_code == 200
        data = response.json

        # Verify BPM and downbeat loaded from annotation
        assert data["bpm"] == 128.5
        assert data["downbeat"] == 0.25

    def test_load_track_no_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns null BPM when no annotation exists."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file without annotation
        audio_file = temp_audio_dir / "unannotated.wav"
        self._create_audio_file(audio_file)

        response = client.get("/api/load/unannotated.wav")

        assert response.status_code == 200
        data = response.json

        # BPM should be null, downbeat defaults to 0.0
        assert data["bpm"] is None
        assert data["downbeat"] == 0.0

    def test_load_track_not_found(self, client, temp_audio_dir, app):
        """Returns 404 for non-existent files."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/load/nonexistent.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_load_track_path_traversal(self, client, temp_audio_dir, app):
        """Returns 400 for path traversal attempts."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        # Try path traversal
        response = client.get("/api/load/../../../etc/passwd")

        assert response.status_code == 400
        assert "error" in response.json

    def test_load_track_corrupted_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Handles corrupted audio files gracefully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create corrupted file (invalid audio data)
        corrupted_file = temp_audio_dir / "corrupted.wav"
        corrupted_file.write_bytes(b"INVALID AUDIO DATA")

        response = client.get("/api/load/corrupted.wav")

        # Should return error (500 or 400)
        assert response.status_code in [400, 500]
        assert "error" in response.json

    @staticmethod
    def _create_audio_file(path: Path, duration: float = 1.0):
        """Create a simple audio file for testing."""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio_data, sample_rate)


class TestSaveAnnotationEndpoint:
    """Test POST /api/save endpoint (CRITICAL)."""

    def test_save_valid_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Saves valid annotation successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test_track.wav"
        self._create_audio_file(audio_file, duration=60.0)

        # Prepare annotation data
        annotation_data = {
            "filename": "test_track.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 15.5, "label": "buildup"},
                {"time": 30.0, "label": "breakdown"},
                {"time": 45.0, "label": "outro"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 200
        data = response.json

        # Verify response
        assert data["success"] is True
        assert "output" in data
        assert data["boundaries_count"] == 4

        # Verify file was created
        ref_dir = temp_annotation_dir / "reference"
        saved_file = ref_dir / "test_track.yaml"
        assert saved_file.exists()

        # Verify saved content
        with open(saved_file) as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["audio"]["bpm"] == 128.0
        assert saved_data["audio"]["downbeat"] == 0.0
        assert len(saved_data["structure"]) == 4

    def test_save_annotation_validates_required_fields(self, client, app):
        """Validates required fields (filename, boundaries)."""
        # Missing filename
        response = client.post(
            "/api/save",
            data=json.dumps({"boundaries": []}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "error" in response.json

        # Missing boundaries
        response = client.post(
            "/api/save",
            data=json.dumps({"filename": "test.wav"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "error" in response.json

    def test_save_annotation_rejects_invalid_labels(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Rejects invalid labels with 400 error."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test.wav"
        self._create_audio_file(audio_file)

        # Invalid label
        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "invalid_label"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 400
        assert "error" in response.json
        assert "invalid" in response.json["error"].lower()

    def test_save_annotation_audio_not_found(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Returns 404 for non-existent audio files."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        annotation_data = {
            "filename": "nonexistent.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 404
        assert "error" in response.json

    def test_save_annotation_creates_reference_directory(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Creates reference directory if missing."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Ensure reference directory doesn't exist
        ref_dir = temp_annotation_dir / "reference"
        if ref_dir.exists():
            import shutil

            shutil.rmtree(ref_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test.wav"
        self._create_audio_file(audio_file)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 200
        assert ref_dir.exists()

    def test_save_annotation_malformed_json(self, client, app):
        """Handles malformed JSON with 400 error."""
        response = client.post("/api/save", data="invalid json{", content_type="application/json")

        assert response.status_code in [400, 415]  # 415 Unsupported Media Type is also valid

    def test_save_annotation_default_values(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Uses default BPM and downbeat values when not provided."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test.wav"
        self._create_audio_file(audio_file)

        # No BPM or downbeat specified
        annotation_data = {
            "filename": "test.wav",
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 200

        # Verify defaults were used
        ref_dir = temp_annotation_dir / "reference"
        saved_file = ref_dir / "test.yaml"
        with open(saved_file) as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["audio"]["bpm"] == 128.0  # Default
        assert saved_data["audio"]["downbeat"] == 0.0  # Default

    def test_save_annotation_empty_boundaries(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles empty boundaries array."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test.wav"
        self._create_audio_file(audio_file)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 200
        data = response.json
        assert data["boundaries_count"] == 0

    @staticmethod
    def _create_audio_file(path: Path, duration: float = 1.0):
        """Create a simple audio file for testing."""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio_data, sample_rate)


class TestAudioFileServing:
    """Test GET /api/audio/<filename> endpoint."""

    def test_serve_audio_file(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Serves audio file successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test.wav"
        self._create_audio_file(audio_file)

        response = client.get("/api/audio/test.wav")

        assert response.status_code == 200
        assert response.content_type.startswith("audio/")
        assert len(response.data) > 0

    def test_serve_audio_missing_file(self, client, temp_audio_dir, app):
        """Returns 404 for missing files."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        response = client.get("/api/audio/nonexistent.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_serve_audio_path_traversal(self, client, temp_audio_dir, app):
        """Blocks path traversal attacks."""
        app.config["AUDIO_DIR"] = temp_audio_dir

        # Try path traversal
        response = client.get("/api/audio/../../../etc/passwd")

        assert response.status_code == 400
        assert "error" in response.json

    @staticmethod
    def _create_audio_file(path: Path, duration: float = 1.0):
        """Create a simple audio file for testing."""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio_data, sample_rate)
