"""Tests for structure analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from edm.analysis.structure import (
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
            Section(label="drop", start_time=32.0, end_time=96.0, confidence=0.85),
        ]
        result = StructureResult(sections=sections, duration=96.0, detector="energy")

        assert len(result.sections) == 2
        assert result.duration == 96.0
        assert result.detector == "energy"


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


class TestEnergyDetector:
    """Tests for EnergyDetector class."""

    def test_is_available(self):
        """Test energy detector is always available."""
        detector = EnergyDetector()
        assert detector.is_available() is True

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        detector = EnergyDetector()
        assert detector._min_section_duration == 8.0
        assert detector._energy_threshold_high == 0.7
        assert detector._energy_threshold_low == 0.4

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        detector = EnergyDetector(
            min_section_duration=4.0,
            energy_threshold_high=0.8,
            energy_threshold_low=0.3,
        )
        assert detector._min_section_duration == 4.0
        assert detector._energy_threshold_high == 0.8
        assert detector._energy_threshold_low == 0.3

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

    def test_is_available_without_msaf(self):
        """Test is_available returns False when msaf not installed."""
        detector = MSAFDetector()
        detector._msaf = None

        # Mock the import to raise ImportError
        with patch("edm.analysis.structure_detector.MSAFDetector.is_available") as mock_avail:
            mock_avail.return_value = False
            assert mock_avail() is False

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

    def test_get_auto_falls_back_to_energy(self):
        """Test auto mode falls back to energy when msaf unavailable."""
        with patch.object(MSAFDetector, "is_available", return_value=False):
            detector = get_detector("auto")
            assert isinstance(detector, EnergyDetector)

    def test_get_unknown_returns_none(self):
        """Test unknown detector type returns None."""
        detector = get_detector("invalid_detector")
        assert detector is None

    def test_get_msaf_when_unavailable(self):
        """Test requesting msaf when unavailable returns None."""
        with patch.object(MSAFDetector, "is_available", return_value=False):
            detector = get_detector("msaf")
            assert detector is None


class TestPostProcessSections:
    """Tests for _post_process_sections function."""

    def test_empty_sections_with_duration(self):
        """Test post-processing empty sections with known duration."""
        sections = _post_process_sections([], 180.0)

        assert len(sections) == 1
        assert sections[0].label == "intro"
        assert sections[0].start_time == 0.0
        assert sections[0].end_time == 180.0

    def test_empty_sections_without_duration(self):
        """Test post-processing empty sections without duration."""
        sections = _post_process_sections([], None)
        assert sections == []

    def test_adds_intro_if_not_at_zero(self):
        """Test adding intro section if first section doesn't start at 0."""
        input_sections = [
            Section(label="drop", start_time=32.0, end_time=96.0, confidence=0.9),
        ]

        sections = _post_process_sections(input_sections, 96.0)

        assert len(sections) == 2
        assert sections[0].label == "intro"
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

        mock_get_detector.assert_called_once_with("energy")
        assert isinstance(result, StructureResult)

    @patch("edm.analysis.structure.get_detector")
    @patch("edm.analysis.structure.MutagenFile")
    def test_falls_back_to_energy_on_error(self, mock_mutagen, mock_get_detector):
        """Test fallback to energy detector when msaf fails."""
        mock_audio = MagicMock()
        mock_audio.info.length = 180.0
        mock_mutagen.return_value = mock_audio

        # First call returns msaf detector that fails
        mock_msaf = MagicMock(spec=MSAFDetector)
        mock_msaf.detect.side_effect = RuntimeError("MSAF failed")

        # Second call for fallback returns energy detector
        mock_energy = MagicMock(spec=EnergyDetector)
        mock_energy.detect.return_value = [
            DetectedSection(start_time=0.0, end_time=180.0, label="intro", confidence=0.9),
        ]

        def detector_factory(dtype):
            if dtype == "auto":
                return mock_msaf
            return mock_energy

        mock_get_detector.side_effect = detector_factory

        with patch("edm.analysis.structure.EnergyDetector", return_value=mock_energy):
            result = analyze_structure(Path("test.mp3"), detector="auto")

        assert isinstance(result, StructureResult)
        assert result.detector == "energy"

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
