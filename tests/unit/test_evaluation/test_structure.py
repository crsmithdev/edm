"""Tests for structure evaluation module."""

from edm.evaluation.evaluators.structure import (
    _calculate_overlap,
    _calculate_structure_metrics,
    load_structure_reference,
)


class TestLoadStructureReference:
    """Tests for load_structure_reference function."""

    def test_loads_valid_csv(self, tmp_path):
        """Test loading a valid CSV file."""
        csv_path = tmp_path / "reference.csv"
        csv_path.write_text(
            "file,label,start,end\n"
            "track1.mp3,intro,0.0,32.0\n"
            "track1.mp3,drop,32.0,96.0\n"
            "track2.mp3,intro,0.0,16.0\n"
        )

        reference = load_structure_reference(csv_path)

        assert len(reference) == 2
        # Reference keys are resolved paths, so we need to find by filename
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]
        assert len(reference[track1_key]) == 2

    def test_sorts_sections_by_start_time(self, tmp_path):
        """Test that sections are sorted by start time."""
        csv_path = tmp_path / "reference.csv"
        csv_path.write_text(
            "file,label,start,end\n"
            "track1.mp3,drop,32.0,96.0\n"
            "track1.mp3,intro,0.0,32.0\n"  # Out of order
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        assert sections[0]["label"] == "intro"
        assert sections[1]["label"] == "drop"


class TestCalculateOverlap:
    """Tests for _calculate_overlap function."""

    def test_no_overlap(self):
        """Test sections with no overlap."""
        s1 = {"start": 0.0, "end": 30.0}
        s2 = {"start": 60.0, "end": 90.0}

        overlap = _calculate_overlap(s1, s2)
        assert overlap == 0.0

    def test_full_overlap(self):
        """Test identical sections (100% overlap)."""
        s1 = {"start": 0.0, "end": 30.0}
        s2 = {"start": 0.0, "end": 30.0}

        overlap = _calculate_overlap(s1, s2)
        assert overlap == 1.0

    def test_partial_overlap(self):
        """Test partial overlap between sections."""
        s1 = {"start": 0.0, "end": 40.0}
        s2 = {"start": 20.0, "end": 60.0}

        overlap = _calculate_overlap(s1, s2)
        # Overlap: 20-40 = 20s
        # Union: 0-60 = 60s
        # IoU = 20/60 = 0.333...
        assert abs(overlap - 1 / 3) < 0.01

    def test_one_contains_other(self):
        """Test when one section contains the other."""
        s1 = {"start": 0.0, "end": 100.0}
        s2 = {"start": 20.0, "end": 40.0}

        overlap = _calculate_overlap(s1, s2)
        # Overlap: 20s (s2 duration)
        # Union: 100s (s1 duration since s2 is inside)
        # IoU = 20/100 = 0.2
        assert abs(overlap - 0.2) < 0.01


class TestCalculateStructureMetrics:
    """Tests for _calculate_structure_metrics function."""

    def test_empty_reference(self):
        """Test with empty reference."""
        metrics = _calculate_structure_metrics([], [{"label": "intro", "start": 0, "end": 30}], 2.0)

        assert metrics["boundary_precision"] == 0.0
        assert metrics["boundary_recall"] == 0.0

    def test_empty_detected(self):
        """Test with empty detected."""
        metrics = _calculate_structure_metrics([{"label": "intro", "start": 0, "end": 30}], [], 2.0)

        assert metrics["boundary_precision"] == 0.0
        assert metrics["boundary_recall"] == 0.0

    def test_perfect_match(self):
        """Test with perfect boundary match."""
        reference = [
            {"label": "intro", "start": 0.0, "end": 30.0},
            {"label": "drop", "start": 30.0, "end": 90.0},
        ]
        detected = [
            {"label": "intro", "start": 0.0, "end": 30.0, "confidence": 0.9},
            {"label": "drop", "start": 30.0, "end": 90.0, "confidence": 0.85},
        ]

        metrics = _calculate_structure_metrics(reference, detected, 2.0)

        assert metrics["boundary_precision"] == 1.0
        assert metrics["boundary_recall"] == 1.0
        assert metrics["boundary_f1"] == 1.0
        assert metrics["label_accuracy"] == 1.0

    def test_boundary_within_tolerance(self):
        """Test boundaries within tolerance are matched."""
        reference = [
            {"label": "intro", "start": 0.0, "end": 30.0},
            {"label": "drop", "start": 30.0, "end": 90.0},
        ]
        detected = [
            {"label": "intro", "start": 0.0, "end": 31.5, "confidence": 0.9},  # Within 2s tolerance
            {"label": "drop", "start": 31.5, "end": 91.0, "confidence": 0.85},
        ]

        metrics = _calculate_structure_metrics(reference, detected, 2.0)

        # Boundaries should match within tolerance
        assert metrics["boundary_recall"] > 0.5

    def test_drop_detection_metrics(self):
        """Test drop-specific detection metrics."""
        reference = [
            {"label": "intro", "start": 0.0, "end": 30.0},
            {"label": "drop", "start": 30.0, "end": 90.0},
            {"label": "drop", "start": 120.0, "end": 180.0},
        ]
        detected = [
            {"label": "intro", "start": 0.0, "end": 30.0, "confidence": 0.9},
            {"label": "drop", "start": 30.0, "end": 90.0, "confidence": 0.85},  # Correct
            {"label": "breakdown", "start": 120.0, "end": 180.0, "confidence": 0.8},  # Missed drop
        ]

        metrics = _calculate_structure_metrics(reference, detected, 2.0)

        # Only one of two drops detected
        assert metrics["drop_recall"] == 0.5
        # One drop detected, one detected as drop
        assert metrics["drop_precision"] == 1.0
