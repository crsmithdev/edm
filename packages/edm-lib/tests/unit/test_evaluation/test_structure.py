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

    def test_bar_based_annotations_with_bpm(self, tmp_path):
        """Test loading bar-based annotations with BPM."""
        csv_path = tmp_path / "reference.csv"
        # At 120 BPM in 4/4: bar 1 = 0s, bar 9 = 16s, bar 17 = 32s
        # Drop is an event (single bar), buildup is a span
        csv_path.write_text(
            "file,label,start_bar,end_bar,bpm\n"
            "track1.mp3,intro,1,9,120\n"
            "track1.mp3,drop,9,,120\n"  # Event - no end_bar
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        # Bar 1 starts at 0s, bar 9 starts at 16s (8 bars * 2s per bar)
        assert sections[0]["label"] == "intro"
        assert abs(sections[0]["start"] - 0.0) < 0.01
        assert abs(sections[0]["end"] - 16.0) < 0.01
        assert sections[0]["is_event"] is False

        assert sections[1]["label"] == "drop"
        assert abs(sections[1]["time"] - 16.0) < 0.01
        assert sections[1]["is_event"] is True

    def test_bar_based_annotations_with_first_downbeat(self, tmp_path):
        """Test loading bar-based annotations with first_downbeat offset."""
        csv_path = tmp_path / "reference.csv"
        # At 120 BPM with first_downbeat=0.5:
        # bar 1 = 0.5s, bar 9 = 16.5s
        csv_path.write_text(
            "file,label,start_bar,end_bar,bpm,first_downbeat\n"
            "track1.mp3,intro,1,9,120,0.5\n"
            "track1.mp3,drop,9,,120,0.5\n"  # Event - no end_bar
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        assert sections[0]["label"] == "intro"
        assert abs(sections[0]["start"] - 0.5) < 0.01
        assert abs(sections[0]["end"] - 16.5) < 0.01

        assert sections[1]["label"] == "drop"
        assert abs(sections[1]["time"] - 16.5) < 0.01
        assert sections[1]["is_event"] is True

    def test_first_downbeat_defaults_to_zero(self, tmp_path):
        """Test that first_downbeat defaults to 0.0 when not provided."""
        csv_path = tmp_path / "reference.csv"
        # Without first_downbeat column
        csv_path.write_text("file,label,start_bar,end_bar,bpm\ntrack1.mp3,intro,1,9,120\n")

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        # Bar 1 at 0s (first_downbeat defaults to 0.0)
        assert abs(sections[0]["start"] - 0.0) < 0.01

    def test_empty_first_downbeat_defaults_to_zero(self, tmp_path):
        """Test that empty first_downbeat value defaults to 0.0."""
        csv_path = tmp_path / "reference.csv"
        csv_path.write_text(
            "file,label,start_bar,end_bar,bpm,first_downbeat\n"
            "track1.mp3,intro,1,9,120,\n"  # Empty first_downbeat
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        # Bar 1 at 0s (first_downbeat defaults to 0.0 when empty)
        assert abs(sections[0]["start"] - 0.0) < 0.01

    def test_mixed_format_prefers_bars_with_bpm(self, tmp_path):
        """Test that bar-based is used when both time and bar columns exist with BPM."""
        csv_path = tmp_path / "reference.csv"
        # Both start/end (time) and start_bar/end_bar with BPM
        csv_path.write_text(
            "file,label,start,end,start_bar,end_bar,bpm,first_downbeat\n"
            "track1.mp3,intro,999,999,1,9,120,0.5\n"  # Time values ignored
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]

        sections = reference[track1_key]
        # Should use bar-based (0.5s) not time-based (999s)
        assert abs(sections[0]["start"] - 0.5) < 0.01

    def test_per_track_first_downbeat(self, tmp_path):
        """Test that different tracks can have different first_downbeat values."""
        csv_path = tmp_path / "reference.csv"
        csv_path.write_text(
            "file,label,start_bar,end_bar,bpm,first_downbeat\n"
            "track1.mp3,intro,1,9,120,0.5\n"
            "track2.mp3,intro,1,9,120,1.0\n"
        )

        reference = load_structure_reference(csv_path)
        track1_key = [k for k in reference.keys() if k.name == "track1.mp3"][0]
        track2_key = [k for k in reference.keys() if k.name == "track2.mp3"][0]

        # track1 has first_downbeat=0.5
        assert abs(reference[track1_key][0]["start"] - 0.5) < 0.01

        # track2 has first_downbeat=1.0
        assert abs(reference[track2_key][0]["start"] - 1.0) < 0.01


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

    def test_event_detection_metrics(self):
        """Test event-specific detection metrics (drops)."""
        reference = [
            {"label": "intro", "start": 0.0, "end": 30.0, "is_event": False},
            {"label": "drop", "time": 30.0, "is_event": True},
            {"label": "drop", "time": 120.0, "is_event": True},
        ]
        detected = [
            {"label": "intro", "start": 0.0, "end": 30.0, "is_event": False, "confidence": 0.9},
            {"label": "drop", "time": 30.0, "is_event": True, "confidence": 0.85},  # Correct
            # Missed second drop
        ]

        metrics = _calculate_structure_metrics(reference, detected, 2.0)

        # Only one of two drops detected
        assert metrics["event_recall"] == 0.5
        # One drop detected, one detected as drop
        assert metrics["event_precision"] == 1.0
