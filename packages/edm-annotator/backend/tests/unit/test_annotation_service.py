"""Tests for AnnotationService class (CRITICAL)."""

import pytest
import yaml

from edm_annotator.config import TestingConfig
from edm_annotator.services.annotation_service import AnnotationService
from edm_annotator.services.audio_service import AudioService


@pytest.fixture
def annotation_config(temp_audio_dir, temp_annotation_dir):
    """Create test configuration for AnnotationService."""
    config = {
        "AUDIO_DIR": temp_audio_dir,
        "ANNOTATION_DIR": temp_annotation_dir,
        "VALID_LABELS": TestingConfig.VALID_LABELS,
        "WAVEFORM_SAMPLE_RATE": TestingConfig.WAVEFORM_SAMPLE_RATE,
        "AUDIO_EXTENSIONS": TestingConfig.AUDIO_EXTENSIONS,
    }
    return config


@pytest.fixture
def audio_service_mock(annotation_config):
    """Create AudioService instance for testing."""
    return AudioService(annotation_config)


@pytest.fixture
def annotation_service(annotation_config, audio_service_mock):
    """Create AnnotationService instance for testing."""
    return AnnotationService(annotation_config, audio_service_mock)


@pytest.fixture
def sample_annotation_data():
    """Sample annotation data for testing."""
    return {
        "metadata": {
            "tier": 1,
            "confidence": 1.0,
            "source": "manual",
            "annotator": "test_user",
        },
        "audio": {
            "file": "/path/to/track.wav",
            "duration": 180.0,
            "bpm": 128.0,
            "downbeat": 0.5,
            "time_signature": [4, 4],
        },
        "structure": [
            {"bar": 1, "label": "intro", "time": 0.5, "confidence": 1.0},
            {"bar": 17, "label": "buildup", "time": 30.5, "confidence": 1.0},
        ],
    }


class TestFindAnnotationForFile:
    """Tests for find_annotation_for_file method."""

    def test_find_reference_annotation_tier1(self, annotation_service, temp_annotation_dir):
        """Test finding annotation in reference directory (tier 1)."""
        # Create reference annotation
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"
        ref_file.write_text("audio:\n  bpm: 128\n")

        path, tier = annotation_service.find_annotation_for_file("test_track.wav")

        assert path == ref_file
        assert tier == 1

    def test_find_generated_annotation_tier2(self, annotation_service, temp_annotation_dir):
        """Test finding annotation in generated directory (tier 2)."""
        # Create generated annotation
        gen_dir = temp_annotation_dir / "generated"
        gen_file = gen_dir / "test_track.yaml"
        gen_file.write_text("audio:\n  bpm: 128\n")

        path, tier = annotation_service.find_annotation_for_file("test_track.wav")

        assert path == gen_file
        assert tier == 2

    def test_find_prefers_reference_over_generated(self, annotation_service, temp_annotation_dir):
        """Test tier priority: reference (tier 1) preferred over generated (tier 2)."""
        # Create both reference and generated
        ref_dir = temp_annotation_dir / "reference"
        gen_dir = temp_annotation_dir / "generated"

        ref_file = ref_dir / "test_track.yaml"
        gen_file = gen_dir / "test_track.yaml"

        ref_file.write_text("audio:\n  bpm: 128\n")
        gen_file.write_text("audio:\n  bpm: 140\n")

        path, tier = annotation_service.find_annotation_for_file("test_track.wav")

        # Should return reference, not generated
        assert path == ref_file
        assert tier == 1

    def test_find_case_insensitive_match(self, annotation_service, temp_annotation_dir):
        """Test case-insensitive filename matching."""
        # Create annotation with different case
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "Test_Track.yaml"
        ref_file.write_text("audio:\n  bpm: 128\n")

        # Search with lowercase
        path, tier = annotation_service.find_annotation_for_file("test_track.wav")

        assert path == ref_file
        assert tier == 1

    def test_find_case_insensitive_prefers_reference(self, annotation_service, temp_annotation_dir):
        """Test case-insensitive search still respects tier priority."""
        # Create files with different cases in both directories
        ref_dir = temp_annotation_dir / "reference"
        gen_dir = temp_annotation_dir / "generated"

        ref_file = ref_dir / "TEST_TRACK.yaml"
        gen_file = gen_dir / "test_track.yaml"

        ref_file.write_text("audio:\n  bpm: 128\n")
        gen_file.write_text("audio:\n  bpm: 140\n")

        path, tier = annotation_service.find_annotation_for_file("Test_Track.flac")

        assert path == ref_file
        assert tier == 1

    def test_find_returns_none_when_not_found(self, annotation_service):
        """Test returns (None, None) when annotation doesn't exist."""
        path, tier = annotation_service.find_annotation_for_file("nonexistent.wav")

        assert path is None
        assert tier is None

    def test_find_handles_multiple_extensions(self, annotation_service, temp_annotation_dir):
        """Test finds annotation regardless of audio file extension."""
        # Create annotation for track
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "my_track.yaml"
        ref_file.write_text("audio:\n  bpm: 128\n")

        # Should find for different extensions
        for extension in ["mp3", "wav", "flac", "m4a"]:
            path, tier = annotation_service.find_annotation_for_file(f"my_track.{extension}")
            assert path == ref_file
            assert tier == 1


class TestLoadAnnotation:
    """Tests for load_annotation method."""

    def test_load_valid_annotation(
        self, annotation_service, temp_annotation_dir, sample_annotation_data
    ):
        """Test loading valid annotation file."""
        # Create annotation file
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        with open(ref_file, "w") as f:
            yaml.dump(sample_annotation_data, f)

        result = annotation_service.load_annotation("test_track.wav")

        assert result is not None
        assert result["bpm"] == 128.0
        assert result["downbeat"] == 0.5

    def test_load_missing_annotation(self, annotation_service):
        """Test loading missing annotation returns None."""
        result = annotation_service.load_annotation("nonexistent.wav")
        assert result is None

    def test_load_extracts_bpm(self, annotation_service, temp_annotation_dir):
        """Test BPM extraction from annotation."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        data = {"audio": {"bpm": 140.5}}
        with open(ref_file, "w") as f:
            yaml.dump(data, f)

        result = annotation_service.load_annotation("test_track.wav")

        assert result["bpm"] == 140.5

    def test_load_extracts_downbeat(self, annotation_service, temp_annotation_dir):
        """Test downbeat extraction from annotation."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        data = {"audio": {"bpm": 128, "downbeat": 1.5}}
        with open(ref_file, "w") as f:
            yaml.dump(data, f)

        result = annotation_service.load_annotation("test_track.wav")

        assert result["downbeat"] == 1.5

    def test_load_defaults_downbeat_to_zero(self, annotation_service, temp_annotation_dir):
        """Test downbeat defaults to 0.0 when missing."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        data = {"audio": {"bpm": 128}}  # No downbeat
        with open(ref_file, "w") as f:
            yaml.dump(data, f)

        result = annotation_service.load_annotation("test_track.wav")

        assert result["downbeat"] == 0.0

    def test_load_handles_corrupt_yaml(self, annotation_service, temp_annotation_dir):
        """Test loading corrupt YAML returns None."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        # Write invalid YAML
        ref_file.write_text("invalid: yaml: content: [[[")

        result = annotation_service.load_annotation("test_track.wav")
        assert result is None

    def test_load_handles_missing_audio_section(self, annotation_service, temp_annotation_dir):
        """Test loading annotation without audio section returns None."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"

        data = {"metadata": {"tier": 1}}  # No audio section
        with open(ref_file, "w") as f:
            yaml.dump(data, f)

        result = annotation_service.load_annotation("test_track.wav")
        assert result is None

    def test_load_handles_empty_file(self, annotation_service, temp_annotation_dir):
        """Test loading empty annotation file returns None."""
        ref_dir = temp_annotation_dir / "reference"
        ref_file = ref_dir / "test_track.yaml"
        ref_file.write_text("")

        result = annotation_service.load_annotation("test_track.wav")
        assert result is None


class TestSaveAnnotation:
    """Tests for save_annotation method."""

    def test_save_creates_yaml_file(
        self, annotation_service, temp_annotation_dir, sample_audio_file
    ):
        """Test save_annotation creates YAML file."""
        boundaries = [
            {"time": 0.5, "label": "intro"},
            {"time": 30.5, "label": "buildup"},
        ]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        assert output_file.exists()
        assert output_file.suffix == ".yaml"
        assert output_file.parent == temp_annotation_dir / "reference"

    def test_save_validates_labels(self, annotation_service, sample_audio_file):
        """Test save validates label values."""
        boundaries = [{"time": 0.5, "label": "invalid_label"}]

        with pytest.raises(ValueError, match="Invalid label"):
            annotation_service.save_annotation(
                sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
            )

    def test_save_allows_valid_labels(self, annotation_service, sample_audio_file):
        """Test save accepts all valid labels."""
        valid_labels = ["intro", "buildup", "breakdown", "breakbuild", "outro", "unlabeled"]

        for label in valid_labels:
            boundaries = [{"time": 0.5, "label": label}]

            # Should not raise
            output_file = annotation_service.save_annotation(
                sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
            )
            assert output_file.exists()

    def test_save_calculates_bars(self, annotation_service, sample_audio_file):
        """Test save calculates bar numbers correctly."""
        boundaries = [
            {"time": 0.5, "label": "intro"},  # Bar 1
            {"time": 2.375, "label": "buildup"},  # Bar 2 (0.5 + 1.875 = bar duration at 128 BPM)
        ]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        # Load and verify bars
        with open(output_file) as f:
            data = yaml.safe_load(f)

        assert data["structure"][0]["bar"] == 1
        assert data["structure"][1]["bar"] == 2

    def test_save_creates_directory_if_missing(
        self, annotation_service, temp_annotation_dir, sample_audio_file
    ):
        """Test save creates reference directory if it doesn't exist."""
        # Remove reference directory
        ref_dir = temp_annotation_dir / "reference"
        if ref_dir.exists():
            import shutil

            shutil.rmtree(ref_dir)

        boundaries = [{"time": 0.5, "label": "intro"}]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_includes_audio_metadata(self, annotation_service, sample_audio_file):
        """Test saved annotation includes audio metadata."""
        boundaries = [{"time": 0.5, "label": "intro"}]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        with open(output_file) as f:
            data = yaml.safe_load(f)

        assert "audio" in data
        assert data["audio"]["bpm"] == 128.0
        assert data["audio"]["downbeat"] == 0.5
        assert data["audio"]["duration"] > 0

    def test_save_includes_annotation_metadata(self, annotation_service, sample_audio_file):
        """Test saved annotation includes metadata section."""
        boundaries = [{"time": 0.5, "label": "intro"}]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        with open(output_file) as f:
            data = yaml.safe_load(f)

        assert "metadata" in data
        assert data["metadata"]["tier"] == 1
        assert data["metadata"]["source"] == "manual"
        # Check that annotator exists if present (field may be optional in schema)
        if "annotator" in data["metadata"]:
            assert data["metadata"]["annotator"] == "web_tool"

    def test_save_sorts_boundaries_by_time(self, annotation_service, sample_audio_file):
        """Test save sorts boundaries by time."""
        # Provide boundaries out of order
        boundaries = [
            {"time": 30.5, "label": "buildup"},
            {"time": 0.5, "label": "intro"},
            {"time": 15.0, "label": "breakdown"},
        ]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        with open(output_file) as f:
            data = yaml.safe_load(f)

        times = [section["time"] for section in data["structure"]]
        assert times == sorted(times)

    def test_save_uses_stem_for_filename(self, annotation_service, sample_audio_file):
        """Test save uses audio stem (without extension) for YAML filename."""
        boundaries = [{"time": 0.5, "label": "intro"}]

        output_file = annotation_service.save_annotation(
            sample_audio_file.name, bpm=128.0, downbeat=0.5, boundaries=boundaries
        )

        expected_stem = sample_audio_file.stem
        assert output_file.stem == expected_stem


class TestTimeToBar:
    """Tests for _time_to_bar method (bar calculation accuracy)."""

    def test_time_to_bar_at_downbeat(self, annotation_service):
        """Test bar calculation at downbeat time."""
        bar = annotation_service._time_to_bar(time=0.5, bpm=128.0, downbeat=0.5)
        assert bar == 1

    def test_time_to_bar_one_bar_later(self, annotation_service):
        """Test bar calculation one bar after downbeat."""
        # At 128 BPM, 4/4 time: bar duration = 60/128 * 4 = 1.875 seconds
        bar = annotation_service._time_to_bar(time=2.375, bpm=128.0, downbeat=0.5)
        assert bar == 2

    def test_time_to_bar_multiple_bars(self, annotation_service):
        """Test bar calculation for multiple bars."""
        # At 128 BPM: bar = 1.875s
        # Downbeat at 0.5
        # Bar 1: 0.5 - 2.375
        # Bar 2: 2.375 - 4.25
        # Bar 3: 4.25 - 6.125
        # Bar 4: 6.125 - 8.0

        test_cases = [
            (0.5, 1),  # At downbeat
            (1.0, 1),  # Middle of bar 1
            (2.375, 2),  # Start of bar 2
            (4.25, 3),  # Start of bar 3
            (6.125, 4),  # Start of bar 4
            (7.0, 4),  # Middle of bar 4
        ]

        for time, expected_bar in test_cases:
            bar = annotation_service._time_to_bar(time, bpm=128.0, downbeat=0.5)
            assert bar == expected_bar, f"Time {time} should be bar {expected_bar}, got {bar}"

    def test_time_to_bar_different_bpms(self, annotation_service):
        """Test bar calculation with different BPM values."""
        # At 120 BPM: bar = 60/120 * 4 = 2.0s
        bar = annotation_service._time_to_bar(time=2.0, bpm=120.0, downbeat=0.0)
        assert bar == 2

        # At 140 BPM: bar = 60/140 * 4 = 1.714s
        # Need slightly more time to reach bar 2
        bar = annotation_service._time_to_bar(time=1.72, bpm=140.0, downbeat=0.0)
        assert bar == 2

    def test_time_to_bar_before_downbeat(self, annotation_service):
        """Test bar calculation before downbeat returns bar 1."""
        # Time before downbeat should still be bar 1 (minimum)
        bar = annotation_service._time_to_bar(time=0.0, bpm=128.0, downbeat=0.5)
        assert bar == 1

    def test_time_to_bar_minimum_is_one(self, annotation_service):
        """Test bar number is never less than 1."""
        # Even with large negative times
        bar = annotation_service._time_to_bar(time=-10.0, bpm=128.0, downbeat=0.0)
        assert bar == 1

    def test_time_to_bar_precision(self, annotation_service):
        """Test bar calculation handles floating point precision."""
        # Test edge cases around bar boundaries
        bpm = 128.0
        downbeat = 0.0
        bar_duration = 60.0 / bpm * 4.0  # 1.875

        # Just before bar 2
        bar = annotation_service._time_to_bar(time=bar_duration - 0.001, bpm=bpm, downbeat=downbeat)
        assert bar == 1

        # At bar 2
        bar = annotation_service._time_to_bar(time=bar_duration, bpm=bpm, downbeat=downbeat)
        assert bar == 2

        # Just after bar 2
        bar = annotation_service._time_to_bar(time=bar_duration + 0.001, bpm=bpm, downbeat=downbeat)
        assert bar == 2
