"""Tests for structure analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from edm.analysis.structure import (
    RawSection,
    Section,
    StructureResult,
    _post_process_sections,
    analyze_structure,
)
from edm.analysis.structure_detector import (
    DetectedSection,
    EnergyDetector,
    MSAFDetector,
    get_detector,
)


class TestSection:
    """Tests for Section dataclass."""

    def test_section_creation(self):
        """Test Section dataclass creation."""
        section = Section(
            label="drop",
            start_time=64.0,
            end_time=128.0,
            confidence=0.95,
        )
        assert section.label == "drop"
        assert section.start_time == 64.0
        assert section.end_time == 128.0
        assert section.confidence == 0.95


class TestStructureResult:
    """Tests for StructureResult dataclass."""

    def test_structure_result_creation(self):
        """Test StructureResult dataclass creation."""
        sections = [
            Section(label="intro", start_time=0.0, end_time=32.0, confidence=0.9),
            Section(label="buildup", start_time=32.0, end_time=96.0, confidence=0.85),
        ]
        events = [(33, "drop"), (65, "drop")]
        raw = [
            RawSection(
                start=0.0, end=32.0, start_bar=0.0, end_bar=17.1, label="intro", confidence=0.9
            ),
            RawSection(
                start=32.0, end=96.0, start_bar=17.1, end_bar=51.2, label="buildup", confidence=0.85
            ),
        ]
        result = StructureResult(
            sections=sections, events=events, raw=raw, duration=96.0, detector="energy"
        )

        assert len(result.sections) == 2
        assert len(result.events) == 2
        assert len(result.raw) == 2
        assert result.duration == 96.0
        assert result.detector == "energy"

    def test_structure_result_with_empty_events(self):
        """Test StructureResult with empty events list."""
        sections = [
            Section(label="intro", start_time=0.0, end_time=32.0, confidence=0.9),
        ]
        raw = [
            RawSection(
                start=0.0, end=32.0, start_bar=None, end_bar=None, label="intro", confidence=0.9
            ),
        ]
        result = StructureResult(
            sections=sections, events=[], raw=raw, duration=32.0, detector="energy"
        )

        assert len(result.sections) == 1
        assert len(result.events) == 0
        assert len(result.raw) == 1


class TestRawSection:
    """Tests for RawSection dataclass."""

    def test_raw_section_creation(self):
        """Test RawSection dataclass creation."""
        raw = RawSection(
            start=0.0,
            end=45.2,
            start_bar=0.0,
            end_bar=24.1,
            label="intro",
            confidence=0.9,
        )
        assert raw.start == 0.0
        assert raw.end == 45.2
        assert raw.start_bar == 0.0
        assert raw.end_bar == 24.1
        assert raw.label == "intro"
        assert raw.confidence == 0.9

    def test_raw_section_without_bars(self):
        """Test RawSection with None bar values."""
        raw = RawSection(
            start=0.0,
            end=30.0,
            start_bar=None,
            end_bar=None,
            label="other",
            confidence=0.5,
        )
        assert raw.start_bar is None
        assert raw.end_bar is None


class TestDetectedSection:
    """Tests for DetectedSection dataclass."""

    def test_detected_section_creation(self):
        """Test DetectedSection dataclass creation."""
        section = DetectedSection(
            start_time=0.0,
            end_time=30.0,
            label="intro",
            confidence=0.8,
        )
        assert section.start_time == 0.0
        assert section.end_time == 30.0
        assert section.label == "intro"
        assert section.confidence == 0.8
        assert section.is_event is False  # Default

    def test_detected_section_as_event(self):
        """Test DetectedSection marked as event."""
        section = DetectedSection(
            start_time=60.0,
            end_time=60.0,
            label="drop",
            confidence=0.9,
            is_event=True,
        )
        assert section.is_event is True
        assert section.label == "drop"


class TestEnergyDetector:
    """Tests for EnergyDetector class."""

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        detector = EnergyDetector()
        assert detector._min_section_duration == 8.0

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        detector = EnergyDetector(
            min_section_duration=4.0,
        )
        assert detector._min_section_duration == 4.0

    def test_merge_short_sections(self):
        """Test that short sections are merged."""
        detector = EnergyDetector(min_section_duration=10.0)

        sections = [
            DetectedSection(start_time=0.0, end_time=20.0, label="intro", confidence=0.9),
            DetectedSection(start_time=20.0, end_time=25.0, label="drop", confidence=0.8),  # Short
            DetectedSection(start_time=25.0, end_time=60.0, label="outro", confidence=0.85),
        ]

        merged = detector._merge_short_sections(sections)

        # Short section should be merged
        assert len(merged) == 2
        assert merged[0].end_time == 25.0  # Extended to cover short section


class TestMSAFDetector:
    """Tests for MSAFDetector class."""

    def test_initialization(self):
        """Test default initialization."""
        detector = MSAFDetector()
        assert detector._boundary_algorithm == "sf"
        assert detector._label_algorithm == "fmc2d"

    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = MSAFDetector(boundary_algorithm="foote", label_algorithm="scluster")
        assert detector._boundary_algorithm == "foote"
        assert detector._label_algorithm == "scluster"

    def test_boundaries_to_sections(self):
        """Test boundary conversion to sections."""
        detector = MSAFDetector()

        boundaries = np.array([0.0, 30.0, 60.0, 120.0])
        labels = np.array([0, 1, 2])

        sections = detector._boundaries_to_sections(boundaries, labels, 120.0)

        assert len(sections) == 3
        assert sections[0].start_time == 0.0
        assert sections[0].end_time == 30.0
        assert sections[1].start_time == 30.0
        assert sections[1].end_time == 60.0
        assert sections[2].start_time == 60.0
        assert sections[2].end_time == 120.0


class TestGetDetector:
    """Tests for get_detector function."""

    def test_get_energy_detector(self):
        """Test getting energy detector."""
        detector = get_detector("energy")
        assert isinstance(detector, EnergyDetector)

    def test_get_auto_returns_msaf(self):
        """Test auto mode returns msaf detector (msaf is required)."""
        detector = get_detector("auto")
        assert isinstance(detector, MSAFDetector)

    def test_get_msaf_detector(self):
        """Test getting msaf detector."""
        detector = get_detector("msaf")
        assert isinstance(detector, MSAFDetector)

    def test_get_unknown_raises_error(self):
        """Test unknown detector type raises ValueError."""
        try:
            get_detector("invalid_detector")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Unknown detector type" in str(e)


class TestPostProcessSections:
    """Tests for _post_process_sections function."""

    def test_empty_sections_with_duration(self):
        """Test post-processing empty sections with known duration."""
        sections = _post_process_sections([], 180.0)

        assert len(sections) == 1
        assert sections[0].label == "segment1"
        assert sections[0].start_time == 0.0
        assert sections[0].end_time == 180.0

    def test_empty_sections_without_duration(self):
        """Test post-processing empty sections without duration."""
        sections = _post_process_sections([], None)
        assert sections == []

    def test_adds_intro_if_not_at_zero(self):
        """Test adding initial section if first section doesn't start at 0."""
        input_sections = [
            Section(label="segment1", start_time=32.0, end_time=96.0, confidence=0.9),
        ]

        sections = _post_process_sections(input_sections, 96.0)

        assert len(sections) == 2
        assert sections[0].label == "segment0"
        assert sections[0].start_time == 0.0
        assert sections[0].end_time == 32.0

    def test_extends_last_section_to_duration(self):
        """Test extending last section to reach duration."""
        input_sections = [
            Section(label="intro", start_time=0.0, end_time=32.0, confidence=0.9),
            Section(label="drop", start_time=32.0, end_time=96.0, confidence=0.85),
        ]

        sections = _post_process_sections(input_sections, 120.0)

        assert sections[-1].end_time == 120.0

    def test_allows_gaps_between_sections(self):
        """Test that gaps between sections are preserved (not filled)."""
        input_sections = [
            Section(label="intro", start_time=0.0, end_time=30.0, confidence=0.9),
            Section(label="drop", start_time=35.0, end_time=96.0, confidence=0.85),  # Gap of 5s
        ]

        sections = _post_process_sections(input_sections, 96.0)

        # Gap should be preserved - sections not extended
        assert sections[0].end_time == 30.0
        assert sections[1].start_time == 35.0

    def test_handles_overlapping_sections(self):
        """Test handling of overlapping sections."""
        input_sections = [
            Section(label="intro", start_time=0.0, end_time=40.0, confidence=0.9),
            Section(label="drop", start_time=30.0, end_time=96.0, confidence=0.85),  # Overlaps
        ]

        sections = _post_process_sections(input_sections, 96.0)

        # Should not have overlaps
        for i in range(len(sections) - 1):
            assert sections[i].end_time <= sections[i + 1].start_time + 0.1

    def test_sorts_by_start_time(self):
        """Test that sections are sorted by start time."""
        input_sections = [
            Section(label="drop", start_time=64.0, end_time=128.0, confidence=0.85),
            Section(label="intro", start_time=0.0, end_time=32.0, confidence=0.9),
            Section(label="buildup", start_time=32.0, end_time=64.0, confidence=0.8),
        ]

        sections = _post_process_sections(input_sections, 128.0)

        for i in range(len(sections) - 1):
            assert sections[i].start_time <= sections[i + 1].start_time


class TestAnalyzeStructure:
    """Tests for analyze_structure function."""

    @patch("edm.analysis.structure.get_detector")
    @patch("edm.analysis.structure.MutagenFile")
    def test_uses_energy_detector_when_specified(self, mock_mutagen, mock_get_detector):
        """Test that energy detector is used when specified."""
        mock_audio = MagicMock()
        mock_audio.info.length = 180.0
        mock_mutagen.return_value = mock_audio

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            DetectedSection(start_time=0.0, end_time=60.0, label="intro", confidence=0.9),
            DetectedSection(start_time=60.0, end_time=180.0, label="drop", confidence=0.85),
        ]
        mock_get_detector.return_value = mock_detector

        result = analyze_structure(Path("test.mp3"), detector="energy")

        mock_get_detector.assert_called_once_with("energy", model_path=None)
        assert isinstance(result, StructureResult)

    @patch("edm.analysis.structure.get_detector")
    @patch("edm.analysis.structure.MutagenFile")
    def test_result_contains_detector_name(self, mock_mutagen, mock_get_detector):
        """Test that result includes detector name."""
        mock_audio = MagicMock()
        mock_audio.info.length = 180.0
        mock_mutagen.return_value = mock_audio

        mock_detector = MagicMock(spec=EnergyDetector)
        mock_detector.detect.return_value = [
            DetectedSection(start_time=0.0, end_time=180.0, label="intro", confidence=0.9),
        ]
        mock_get_detector.return_value = mock_detector

        result = analyze_structure(Path("test.mp3"), detector="energy")

        assert result.detector == "energy"
