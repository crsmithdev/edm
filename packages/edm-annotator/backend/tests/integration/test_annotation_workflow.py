"""End-to-end annotation workflow integration tests.

Tests complete annotation workflows from track loading through
saving and reloading to verify data persistence.
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


class TestAnnotationWorkflow:
    """Test end-to-end annotation workflows."""

    def test_full_annotation_cycle(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Test full cycle: load track -> save annotation -> reload -> verify persistence.

        This tests the most common user workflow:
        1. User loads a track (GET /api/load/<filename>)
        2. User annotates boundaries in the UI
        3. User saves annotation (POST /api/save)
        4. User reloads the track
        5. Annotation data should be restored
        """
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create test audio file
        audio_file = temp_audio_dir / "workflow_test.wav"
        self._create_audio_file(audio_file, duration=120.0)

        # Step 1: Load track initially (no annotation)
        response = client.get("/api/load/workflow_test.wav")
        assert response.status_code == 200
        initial_data = response.json

        # Verify no annotation exists yet
        assert initial_data["bpm"] is None
        assert initial_data["downbeat"] == 0.0

        # Step 2: Create annotation data (simulating user annotation)
        annotation_data = {
            "filename": "workflow_test.wav",
            "bpm": 140.0,
            "downbeat": 0.15,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 20.5, "label": "buildup"},
                {"time": 45.0, "label": "breakdown"},
                {"time": 80.0, "label": "buildup"},
                {"time": 100.0, "label": "outro"},
            ],
        }

        # Step 3: Save annotation
        response = client.post(
            "/api/save", data=json.dumps(annotation_data), content_type="application/json"
        )
        assert response.status_code == 200
        save_response = response.json
        assert save_response["success"] is True
        assert save_response["boundaries_count"] == 5

        # Step 4: Reload track - annotation should be restored
        response = client.get("/api/load/workflow_test.wav")
        assert response.status_code == 200
        reloaded_data = response.json

        # Verify annotation was persisted
        assert reloaded_data["bpm"] == 140.0
        assert reloaded_data["downbeat"] == 0.15
        assert reloaded_data["filename"] == "workflow_test.wav"

        # Step 5: Verify saved YAML file structure
        ref_dir = temp_annotation_dir / "reference"
        saved_file = ref_dir / "workflow_test.yaml"
        assert saved_file.exists()

        with open(saved_file) as f:
            saved_yaml = yaml.safe_load(f)

        # Verify YAML structure
        assert "metadata" in saved_yaml
        assert "audio" in saved_yaml
        assert "structure" in saved_yaml

        assert saved_yaml["audio"]["bpm"] == 140.0
        assert saved_yaml["audio"]["downbeat"] == 0.15
        assert len(saved_yaml["structure"]) == 5

        # Verify boundaries were saved with correct labels
        labels = [section["label"] for section in saved_yaml["structure"]]
        assert labels == ["intro", "buildup", "breakdown", "buildup", "outro"]

    def test_overwrite_existing_annotation(self, client, temp_audio_dir, temp_annotation_dir, app):
        """Test overwriting an existing annotation.

        Workflow:
        1. Save initial annotation
        2. Modify boundaries
        3. Save again
        4. Verify latest annotation is loaded
        """
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "overwrite_test.wav"
        self._create_audio_file(audio_file, duration=60.0)

        # Save initial annotation
        initial_annotation = {
            "filename": "overwrite_test.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 30.0, "label": "outro"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(initial_annotation), content_type="application/json"
        )
        assert response.status_code == 200

        # Verify initial annotation was saved
        response = client.get("/api/load/overwrite_test.wav")
        assert response.status_code == 200
        assert response.json["bpm"] == 128.0

        # Save updated annotation with different BPM and boundaries
        updated_annotation = {
            "filename": "overwrite_test.wav",
            "bpm": 140.0,
            "downbeat": 0.25,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 15.0, "label": "buildup"},
                {"time": 30.0, "label": "breakdown"},
                {"time": 45.0, "label": "outro"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(updated_annotation), content_type="application/json"
        )
        assert response.status_code == 200

        # Reload and verify updated values
        response = client.get("/api/load/overwrite_test.wav")
        assert response.status_code == 200
        reloaded = response.json

        assert reloaded["bpm"] == 140.0
        assert reloaded["downbeat"] == 0.25

        # Verify YAML has 4 boundaries now
        ref_dir = temp_annotation_dir / "reference"
        saved_file = ref_dir / "overwrite_test.yaml"
        with open(saved_file) as f:
            saved_data = yaml.safe_load(f)

        assert len(saved_data["structure"]) == 4

    def test_track_list_reflects_saved_annotations(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Test that track list correctly reflects annotation status after saving.

        Workflow:
        1. List tracks (no annotations)
        2. Save annotation for one track
        3. List tracks again
        4. Verify has_reference flag is updated
        """
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create multiple audio files
        audio_files = ["track_a.wav", "track_b.wav", "track_c.wav"]
        for filename in audio_files:
            self._create_audio_file(temp_audio_dir / filename)

        # Step 1: List tracks before annotation
        response = client.get("/api/tracks")
        assert response.status_code == 200
        tracks_before = response.json

        # All tracks should have no annotations
        assert len(tracks_before) == 3
        for track in tracks_before:
            assert track["has_reference"] is False
            assert track["has_generated"] is False

        # Step 2: Save annotation for track_b
        annotation = {
            "filename": "track_b.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [{"time": 0.0, "label": "intro"}],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation), content_type="application/json"
        )
        assert response.status_code == 200

        # Step 3: List tracks after annotation
        response = client.get("/api/tracks")
        assert response.status_code == 200
        tracks_after = response.json

        # Find track_b in the list
        track_b = next(t for t in tracks_after if t["filename"] == "track_b.wav")
        assert track_b["has_reference"] is True
        assert track_b["has_generated"] is False

        # Other tracks should still have no annotations
        for track in tracks_after:
            if track["filename"] != "track_b.wav":
                assert track["has_reference"] is False
                assert track["has_generated"] is False

    def test_multiple_tracks_independent_annotations(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Test that multiple tracks can be annotated independently.

        Ensures annotations don't interfere with each other.
        """
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create two tracks
        track1_file = temp_audio_dir / "track1.wav"
        track2_file = temp_audio_dir / "track2.wav"
        self._create_audio_file(track1_file, duration=60.0)
        self._create_audio_file(track2_file, duration=90.0)

        # Annotate track1
        annotation1 = {
            "filename": "track1.wav",
            "bpm": 125.0,
            "downbeat": 0.1,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 30.0, "label": "outro"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation1), content_type="application/json"
        )
        assert response.status_code == 200

        # Annotate track2 with different values
        annotation2 = {
            "filename": "track2.wav",
            "bpm": 140.0,
            "downbeat": 0.25,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 20.0, "label": "buildup"},
                {"time": 60.0, "label": "breakdown"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation2), content_type="application/json"
        )
        assert response.status_code == 200

        # Load track1 - should have its own annotation
        response = client.get("/api/load/track1.wav")
        assert response.status_code == 200
        track1_data = response.json
        assert track1_data["bpm"] == 125.0
        assert track1_data["downbeat"] == 0.1

        # Load track2 - should have its own annotation
        response = client.get("/api/load/track2.wav")
        assert response.status_code == 200
        track2_data = response.json
        assert track2_data["bpm"] == 140.0
        assert track2_data["downbeat"] == 0.25

        # Verify YAML files are separate
        ref_dir = temp_annotation_dir / "reference"
        assert (ref_dir / "track1.yaml").exists()
        assert (ref_dir / "track2.yaml").exists()

        with open(ref_dir / "track1.yaml") as f:
            track1_yaml = yaml.safe_load(f)
        with open(ref_dir / "track2.yaml") as f:
            track2_yaml = yaml.safe_load(f)

        assert len(track1_yaml["structure"]) == 2
        assert len(track2_yaml["structure"]) == 3

    def test_annotation_with_all_valid_labels(
        self, client, temp_audio_dir, temp_annotation_dir, app
    ):
        """Test annotation using all valid label types.

        Ensures all valid labels are properly saved and loaded.
        """
        reinitialize_services(app, temp_audio_dir, temp_annotation_dir)

        # Create audio file
        audio_file = temp_audio_dir / "all_labels.wav"
        self._create_audio_file(audio_file, duration=180.0)

        # Create annotation with all valid labels
        annotation = {
            "filename": "all_labels.wav",
            "bpm": 128.0,
            "downbeat": 0.0,
            "boundaries": [
                {"time": 0.0, "label": "intro"},
                {"time": 30.0, "label": "buildup"},
                {"time": 60.0, "label": "breakdown"},
                {"time": 90.0, "label": "breakbuild"},
                {"time": 120.0, "label": "unlabeled"},
                {"time": 150.0, "label": "outro"},
            ],
        }

        response = client.post(
            "/api/save", data=json.dumps(annotation), content_type="application/json"
        )
        assert response.status_code == 200

        # Reload and verify
        response = client.get("/api/load/all_labels.wav")
        assert response.status_code == 200

        # Verify YAML structure
        ref_dir = temp_annotation_dir / "reference"
        with open(ref_dir / "all_labels.yaml") as f:
            saved_data = yaml.safe_load(f)

        labels = [section["label"] for section in saved_data["structure"]]
        expected_labels = ["intro", "buildup", "breakdown", "breakbuild", "unlabeled", "outro"]
        assert labels == expected_labels

    @staticmethod
    def _create_audio_file(path: Path, duration: float = 1.0):
        """Create a simple audio file for testing."""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio_data, sample_rate)
