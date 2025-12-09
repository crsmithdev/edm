"""Tests for validation result dataclasses."""

from edm.analysis.validation.results import (
    AlignmentError,
    CorrectionProposal,
    CrossValidationResult,
    ErrorPattern,
    ValidationResult,
)


def test_alignment_error_creation():
    """Test creating AlignmentError."""
    error = AlignmentError(
        boundary_index=0,
        expected_bar=8.0,
        actual_bar=8.34,
        offset_bars=0.34,
        offset_seconds=0.15,
        confidence=0.9,
    )

    assert error.boundary_index == 0
    assert error.expected_bar == 8.0
    assert error.actual_bar == 8.34
    assert error.offset_bars == 0.34
    assert error.offset_seconds == 0.15
    assert error.confidence == 0.9


def test_correction_proposal_creation():
    """Test creating CorrectionProposal."""
    correction = CorrectionProposal(
        action="quantize_structure",
        target="structure",
        old_value="8.34",
        new_value="8.0",
        confidence=0.85,
        reason="BPM confidence (0.95) > structure confidence (0.75)",
    )

    assert correction.action == "quantize_structure"
    assert correction.target == "structure"
    assert correction.old_value == "8.34"
    assert correction.new_value == "8.0"
    assert correction.confidence == 0.85
    assert "BPM confidence" in correction.reason


def test_validation_result_defaults():
    """Test ValidationResult with default fields."""
    result = ValidationResult(
        validator_name="beat_structure",
        is_valid=True,
        confidence=0.9,
        uncertainty=0.1,
        pattern=ErrorPattern.NO_ERROR,
    )

    assert result.validator_name == "beat_structure"
    assert result.is_valid is True
    assert result.confidence == 0.9
    assert result.uncertainty == 0.1
    assert result.pattern == ErrorPattern.NO_ERROR
    assert result.errors == []
    assert result.corrections == []
    assert result.message == ""


def test_validation_result_with_errors():
    """Test ValidationResult with errors and corrections."""
    error = AlignmentError(
        boundary_index=0,
        expected_bar=8.0,
        actual_bar=8.34,
        offset_bars=0.34,
        offset_seconds=0.15,
        confidence=0.9,
    )

    correction = CorrectionProposal(
        action="quantize_structure",
        target="structure",
        old_value="8.34",
        new_value="8.0",
        confidence=0.85,
        reason="Trust BPM over structure",
    )

    result = ValidationResult(
        validator_name="beat_structure",
        is_valid=False,
        confidence=0.8,
        uncertainty=0.4,
        pattern=ErrorPattern.SINGLE_BOUNDARY_ERROR,
        errors=[error],
        corrections=[correction],
        message="1 boundary misaligned",
    )

    assert result.is_valid is False
    assert len(result.errors) == 1
    assert len(result.corrections) == 1
    assert result.pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR


def test_cross_validation_result_defaults():
    """Test CrossValidationResult with defaults."""
    result = CrossValidationResult(
        is_valid=True,
        confidence=0.9,
        uncertainty=0.1,
    )

    assert result.is_valid is True
    assert result.confidence == 0.9
    assert result.uncertainty == 0.1
    assert result.validators == []
    assert result.needs_review is False
    assert result.summary == ""


def test_cross_validation_result_with_validators():
    """Test CrossValidationResult with multiple validators."""
    validator1 = ValidationResult(
        validator_name="beat_structure",
        is_valid=True,
        confidence=0.9,
        uncertainty=0.1,
        pattern=ErrorPattern.NO_ERROR,
    )

    validator2 = ValidationResult(
        validator_name="downbeat_structure",
        is_valid=False,
        confidence=0.7,
        uncertainty=0.5,
        pattern=ErrorPattern.DOWNBEAT_PHASE_ERROR,
        message="Downbeat phase error detected",
    )

    result = CrossValidationResult(
        is_valid=False,
        confidence=0.8,
        uncertainty=0.5,
        validators=[validator1, validator2],
        needs_review=True,
        summary="1/2 validators passed, manual review recommended",
    )

    assert result.is_valid is False
    assert len(result.validators) == 2
    assert result.needs_review is True
    assert "manual review" in result.summary


def test_error_pattern_enum():
    """Test ErrorPattern enum values."""
    assert ErrorPattern.BPM_SYSTEMATIC_OFFSET.value == "bpm_systematic_offset"
    assert ErrorPattern.DOWNBEAT_PHASE_ERROR.value == "downbeat_phase_error"
    assert ErrorPattern.CONVERTER_BUG.value == "converter_bug"
    assert ErrorPattern.NO_ERROR.value == "no_error"
