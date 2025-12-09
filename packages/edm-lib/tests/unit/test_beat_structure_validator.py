"""Tests for beat/structure alignment validator."""

from edm.analysis.bpm import BPMResult
from edm.analysis.structure import Section, StructureResult
from edm.analysis.validation.beat_structure import BeatStructureValidator
from edm.analysis.validation.results import ErrorPattern


def test_beat_structure_validator_perfect_alignment():
    """Test validator with perfectly aligned boundaries."""
    # 128 BPM, 4/4 time, downbeat at 0.5s
    # Bar 1 starts at 0.5s, bar 2 at 2.375s, bar 3 at 4.25s, bar 4 at 6.125s, bar 5 at 8.0s
    # Bar duration = 60/128 * 4 = 1.875s
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # Structure with boundaries exactly on bar lines
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=8.0,
                confidence=0.9,
                start_bar=1.0,
                end_bar=4.0,
            ),
            Section(
                label="buildup",
                start_time=8.0,
                end_time=14.75,
                confidence=0.85,
                start_bar=5.0,
                end_bar=8.0,
            ),
        ],
        events=[],
        raw=[],
        detector="msaf",
        duration=20.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = BeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is True
    assert result.pattern == ErrorPattern.NO_ERROR
    assert len(result.errors) == 0
    assert result.confidence > 0.9


def test_beat_structure_validator_converter_bug():
    """Test detection of converter bug (all bars at 1)."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # Simulated converter bug: all start_bar = 1
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=7.625,
                confidence=0.8,
                start_bar=1.0,
                end_bar=4.0,
            ),
            Section(
                label="buildup",
                start_time=7.625,
                end_time=14.75,
                confidence=0.8,
                start_bar=1.0,
                end_bar=8.0,
            ),
            Section(
                label="drop",
                start_time=14.75,
                end_time=28.625,
                confidence=0.8,
                start_bar=1.0,
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

    validator = BeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert result.pattern == ErrorPattern.CONVERTER_BUG
    assert "Converter bug" in result.message
    assert len(result.errors) >= 2  # At least boundaries 2 and 3 should be errors


def test_beat_structure_validator_single_boundary_error():
    """Test detection of single boundary error."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,
        source="computed",
        method="beat_this",
    )

    # One boundary misaligned (buildup), others correct
    # Bar 5 = 8.0s, Bar 9 = 15.5s
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
                start_time=8.2,
                end_time=15.5,
                confidence=0.7,
                start_bar=5.2,
                end_bar=9.0,
            ),  # Misaligned start
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

    validator = BeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert result.pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR
    assert len(result.errors) == 1
    assert result.errors[0].boundary_index == 1


def test_beat_structure_validator_insufficient_confidence():
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
                end_time=7.625,
                confidence=0.9,
                start_bar=1.0,
                end_bar=4.0,
            ),
        ],
        events=[],
        raw=[],
        detector="energy",
        duration=10.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = BeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    # Should not be applicable due to low BPM confidence
    assert result.is_valid is False
    assert result.pattern == ErrorPattern.UNKNOWN
    assert "Insufficient data quality" in result.message


def test_beat_structure_validator_proposes_corrections():
    """Test that validator proposes corrections when BPM confidence is higher."""
    bpm_result = BPMResult(
        bpm=128.0,
        confidence=0.95,  # High BPM confidence
        source="computed",
        method="beat_this",
    )

    # Structure with lower confidence
    structure_result = StructureResult(
        sections=[
            Section(
                label="intro",
                start_time=0.5,
                end_time=7.625,
                confidence=0.7,
                start_bar=1.0,
                end_bar=4.0,
            ),
            Section(
                label="buildup",
                start_time=8.0,
                end_time=14.75,
                confidence=0.7,
                start_bar=5.2,
                end_bar=8.0,
            ),  # Misaligned
        ],
        events=[],
        raw=[],
        detector="energy",
        duration=20.0,
        downbeat=0.5,
        time_signature=(4, 4),
    )

    validator = BeatStructureValidator()
    result = validator.validate(bpm_result, structure_result)

    assert result.is_valid is False
    assert len(result.corrections) > 0
    assert result.corrections[0].action == "quantize_structure"
    assert "BPM confidence" in result.corrections[0].reason
