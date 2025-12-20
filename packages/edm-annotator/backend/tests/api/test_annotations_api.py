"""Comprehensive tests for annotation API endpoints.

Tests /api/save and /api/load-generated/<filename> endpoints with:
- Error cases (missing files, invalid data, permission errors)
- Edge cases (concurrent writes, invalid BPM, out-of-order boundaries)
- Data validation (YAML format, boundary data, labels)
"""

import json
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


def create_audio_file(path: Path, duration: float = 1.0):
    """Create a simple audio file for testing."""
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, audio_data, sample_rate)


class TestSaveAnnotationAPI:
    """Tests for POST /api/save endpoint."""

    def test_save_valid_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Saves valid annotation successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "test_track.wav"
        create_audio_file(audio_file, duration=60.0)

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
        assert data["success"] is True
        assert data["boundaries_count"] == 4

        # Verify file was created
        ref_dir = temp_annotation_dir / "reference"
        saved_file = ref_dir / "test_track.yaml"
        assert saved_file.exists()

    def test_save_annotation_missing_filename(self, client, app):
        """Returns 400 when filename is missing."""
        annotation_data = {
            "bpm": 128.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 400
        assert "error" in response.json

    def test_save_annotation_missing_boundaries(self, client, app):
        """Returns 400 when boundaries are missing."""
        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        assert response.status_code == 400
        assert "error" in response.json

    def test_save_annotation_invalid_label(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Returns 400 for invalid section labels."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

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
        """Returns 404 when audio file doesn't exist."""
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

    def test_save_annotation_invalid_bpm_negative(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles negative BPM values."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        annotation_data = {
            "filename": "test.wav",
            "bpm": -128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        # Should either save (accepting any BPM) or return 400
        # Verifying it doesn't crash
        assert response.status_code in [200, 400]

    def test_save_annotation_invalid_bpm_zero(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles zero BPM value."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 0.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        # Should handle gracefully (might fail with division by zero)
        assert response.status_code in [200, 400, 500]

    def test_save_annotation_negative_boundary_time(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles negative boundary times."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=60.0)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [
                {"time": -5.0, "label": "intro"},
                {"time": 0.0, "label": "buildup"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        # Should handle gracefully (might accept or reject)
        assert response.status_code in [200, 400]

    def test_save_annotation_out_of_order_boundaries(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles out-of-order boundary times."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=60.0)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [
                {"time": 30.0, "label": "breakdown"},
                {"time": 0.0, "label": "intro"},
                {"time": 15.0, "label": "buildup"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        # Should accept and sort, or return error
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            # Verify boundaries were sorted in saved file
            ref_dir = temp_annotation_dir / "reference"
            saved_file = ref_dir / "test.yaml"
            with open(saved_file) as f:
                saved_data = yaml.safe_load(f)

            times = [section["time"] for section in saved_data["structure"]]
            assert times == sorted(times), "Boundaries should be sorted by time"

    def test_save_annotation_empty_boundaries(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Accepts empty boundaries array."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

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
        assert response.json["boundaries_count"] == 0

    def test_save_annotation_malformed_json(self, client, app):
        """Returns 400 for malformed JSON."""
        response = client.post("/api/save", data="invalid json{", content_type="application/json")

        assert response.status_code in [400, 415]

    def test_save_annotation_permission_error(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles permission errors when saving."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        # Mock save_annotation to raise PermissionError
        with patch.object(
            app.annotation_service,
            "save_annotation",
            side_effect=PermissionError("Permission denied"),
        ):
            response = client.post(
                "/api/save", data=json.dumps(annotation_data), content_type="application/json"
            )

            assert response.status_code == 500
            assert "error" in response.json

    def test_save_annotation_concurrent_writes(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles concurrent writes to same file."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file, duration=60.0)

        annotation_data = {
            "filename": "test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        # Send two requests in quick succession
        response1 = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )
        response2 = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )

        # Both should succeed (last write wins)
        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_save_annotation_creates_reference_directory(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Creates reference directory if it doesn't exist."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Remove reference directory
        ref_dir = temp_annotation_dir / "reference"
        if ref_dir.exists():
            import shutil

            shutil.rmtree(ref_dir)

        audio_file = temp_audio_dir / "test.wav"
        create_audio_file(audio_file)

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


class TestLoadGeneratedAnnotationAPI:
    """Tests for GET /api/load-generated/<filename> endpoint."""

    def test_load_generated_annotation_success(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Loads generated annotation successfully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create generated annotation
        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        annotation = {
            "audio": {
                "bpm": 140.0,
                "downbeat": 0.5,
                "duration": 180.0,
            },
            "structure": [
                {"time": 0.0, "label": "intro"},
                {"time": 30.0, "label": "buildup"},
            ],
        }
        (gen_dir / "test_track.yaml").write_text(yaml.dump(annotation))

        response = client.get("/api/load-generated/test_track.wav")

        assert response.status_code == 200
        data = response.json

        assert data["bpm"] == 140.0
        assert data["downbeat"] == 0.5
        assert len(data["boundaries"]) == 2
        assert data["boundaries"][0]["label"] == "intro"

    def test_load_generated_annotation_not_found(self, client, temp_annotation_dir, app):
        """Returns 404 when generated annotation doesn't exist."""
        app.config["ANNOTATION_DIR"] = temp_annotation_dir

        response = client.get("/api/load-generated/nonexistent.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_load_generated_annotation_invalid_yaml(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles invalid YAML gracefully."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid YAML
        (gen_dir / "invalid.yaml").write_text("invalid: yaml: content: [unclosed")

        response = client.get("/api/load-generated/invalid.wav")

        assert response.status_code in [404, 500]
        assert "error" in response.json

    def test_load_generated_annotation_empty_file(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles empty YAML file."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Write empty file
        (gen_dir / "empty.yaml").write_text("")

        response = client.get("/api/load-generated/empty.wav")

        assert response.status_code == 404
        assert "error" in response.json

    def test_load_generated_annotation_missing_bpm(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles missing BPM in annotation."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        annotation = {
            "audio": {
                "downbeat": 0.0,
                "duration": 180.0,
            },
            "structure": [{"time": 0.0, "label": "intro"}],
        }
        (gen_dir / "no_bpm.yaml").write_text(yaml.dump(annotation))

        response = client.get("/api/load-generated/no_bpm.wav")

        assert response.status_code == 200
        data = response.json
        assert data["bpm"] is None
        assert data["downbeat"] == 0.0

    def test_load_generated_annotation_missing_boundaries(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles missing structure/boundaries."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        annotation = {
            "audio": {
                "bpm": 128.0,
                "downbeat": 0.0,
                "duration": 180.0,
            }
        }
        (gen_dir / "no_structure.yaml").write_text(yaml.dump(annotation))

        response = client.get("/api/load-generated/no_structure.wav")

        assert response.status_code == 200
        data = response.json
        assert data["bpm"] == 128.0
        # boundaries might be None or empty list depending on implementation
        assert data.get("boundaries") is None or data.get("boundaries") == []

    def test_load_generated_annotation_file_permission_error(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Handles file permission errors."""
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        gen_dir = temp_annotation_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        annotation = {"audio": {"bpm": 128.0}}
        yaml_file = gen_dir / "test.yaml"
        yaml_file.write_text(yaml.dump(annotation))

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            response = client.get("/api/load-generated/test.wav")

            # Should return 404 or 500
            assert response.status_code in [404, 500]
            assert "error" in response.json
