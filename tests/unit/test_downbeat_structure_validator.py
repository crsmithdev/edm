"""Tests for downbeat/structure alignment validator."""

from edm.analysis.bpm import BPMResult
from edm.analysis.structure import Section, StructureResult
from edm.analysis.validation.downbeat_structure import DownbeatStructureValidator
from edm.analysis.validation.results import ErrorPattern


def test_downbeat_structure_validator_perfect_alignment():
    """Test validator with perfectly aligned boundaries on downbeats."""
    # 128 BPM, downbeat at 0.5s
    # Bar duration = 60/128 * 4 = 1.875s
    # Downbeats: 0.5s, 2.375s, 4.25s, 6.125s, 8.0s, 9.875s, etc.
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # Structure boundaries exactly on downbeats
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=8.0,
                confidence=0.9,
                start_bar=1.0,
                end_bar=5.0,
            ),
            Section(
                label="buildup",
                start_time=8.0,
                end_time=15.5,
                confidence=0.85,
                start_bar=5.0,
                end_bar=9.0,
            ),
            Section(
                label="drop",
                start_time=15.5,
                end_time=28.625,
                confidence=0.9,
                start_bar=9.0,
                end_bar=16.0,
            ),
        ],
        events=[],
        raw=[],
        detector="msaf",
        duration=30.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = DownbeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is True
    assert result.pattern == ErrorPattern.NO_ERROR
    assert len(result.errors) == 0
    assert result.confidence > 0.9


def test_downbeat_structure_validator_phase_error():
    """Test detection of downbeat phase error."""
    # 128 BPM, downbeat at 0.5s
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # All boundaries offset by ~0.5 bars (2 beats, phase error)
    # Phase offset = 60/128 * 2 = 0.9375s
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=1.4375,
                end_time=8.9375,
                confidence=0.8,
                start_bar=1.5,
                end_bar=5.5,
            ),
            Section(
                label="buildup",
                start_time=8.9375,
                end_time=16.4375,
                confidence=0.8,
                start_bar=5.5,
                end_bar=9.5,
            ),
            Section(
                label="drop",
                start_time=16.4375,
                end_time=29.5625,
                confidence=0.8,
                start_bar=9.5,
                end_bar=16.5,
            ),
        ],
        events=[],
        raw=[],
        detector="msaf",
        duration=30.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = DownbeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert result.pattern == ErrorPattern.DOWNBEAT_PHASE_ERROR
    assert "phase error" in result.message.lower()


def test_downbeat_structure_validator_first_downbeat_misaligned():
    """Test detection of first downbeat misalignment."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # First section starts significantly before downbeat
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.0,
                end_time=8.0,
                confidence=0.9,
                start_bar=0.73,
                end_bar=5.0,
            ),  # Starts early
            Section(
                label="buildup",
                start_time=8.0,
                end_time=15.5,
                confidence=0.9,
                start_bar=5.0,
                end_bar=9.0,
            ),
        ],
        events=[],
        raw=[],
        detector="msaf",
        duration=20.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = DownbeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert result.pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR
    assert "first" in result.message.lower()


def test_downbeat_structure_validator_single_boundary_error():
    """Test detection of single boundary misaligned."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # One boundary misaligned, others correct
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=8.0,
                confidence=0.9,
                start_bar=1.0,
                end_bar=5.0,
            ),
            Section(
                label="buildup",
                start_time=8.5,
                end_time=15.5,
                confidence=0.7,
                start_bar=5.13,
                end_bar=9.0,
            ),  # Off by 0.5s
            Section(
                label="drop",
                start_time=15.5,
                end_time=28.625,
                confidence=0.9,
                start_bar=9.0,
                end_bar=16.0,
            ),
        ],
        events=[],
        raw=[],
        detector="msaf",
        duration=30.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = DownbeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert result.pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR
    assert len(result.errors) == 1
    assert result.errors[0].boundary_index == 1


def test_downbeat_structure_validator_insufficient_confidence():
    """Test validator with low confidence inputs."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.2,  # Low confidence
        source="computed",
        method="librosa",
    )

    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=8.0,
                confidence=0.9,
                start_bar=1.0,
                end_bar=5.0,
            ),
        ],
        events=[],
        raw=[],
        detector="energy",
        duration=10.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = DownbeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    # Should not be applicable due to low BPM confidence
    assert result.is_valid is False
    assert result.pattern == ErrorPattern.UNKNOWN
    assert "Insufficient data quality" in result.message
