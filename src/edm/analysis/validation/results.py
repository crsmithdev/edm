"""Result dataclasses for cross-validation."""

from dataclasses import dataclass, field
from enum import Enum


class ErrorPattern(Enum):
    """Classification of alignment error patterns."""

    BPM_SYSTEMATIC_OFFSET = "bpm_systematic_offset"
    DOWNBEAT_PHASE_ERROR = "downbeat_phase_error"
    TIME_SIGNATURE_MISMATCH = "time_signature_mismatch"
    SINGLE_BOUNDARY_ERROR = "single_boundary_error"
    PROGRESSIVE_DRIFT = "progressive_drift"
    CONVERTER_BUG = "converter_bug"
    NO_ERROR = "no_error"
    UNKNOWN = "unknown"


@dataclass
class AlignmentError:
    """Single boundary alignment error.

    Attributes:
        boundary_index: Index of the structure boundary
        expected_bar: Expected bar position from structure
        actual_bar: Actual bar position from beat grid
        offset_bars: Difference in bars (expected - actual)
        offset_seconds: Difference in seconds
        confidence: Confidence that this is an error (0-1)
    """

    boundary_index: int
    expected_bar: float
    actual_bar: float
    offset_bars: float
    offset_seconds: float
    confidence: float


@dataclass
class CorrectionProposal:
    """Proposed correction for an alignment error.

    Attributes:
        action: Suggested action (quantize_structure, correct_bpm, flag_conflict, etc.)
        target: What to correct (bpm, structure, downbeat, etc.)
        old_value: Current value
        new_value: Proposed corrected value
        confidence: Confidence in this correction (0-1)
        reason: Human-readable explanation
    """

    action: str
    target: str
    old_value: float | str
    new_value: float | str
    confidence: float
    reason: str


@dataclass
class ValidationResult:
    """Result from a single validator.

    Attributes:
        validator_name: Name of the validator
        is_valid: Whether validation passed
        confidence: Validator's confidence in its result (0-1)
        uncertainty: Measure of signal disagreement (0-1)
        pattern: Detected error pattern
        errors: List of alignment errors found
        corrections: List of proposed corrections
        message: Human-readable summary
    """

    validator_name: str
    is_valid: bool
    confidence: float
    uncertainty: float
    pattern: ErrorPattern
    errors: list[AlignmentError] = field(default_factory=list)
    corrections: list[CorrectionProposal] = field(default_factory=list)
    message: str = ""


@dataclass
class CrossValidationResult:
    """Overall cross-validation result.

    Attributes:
        is_valid: Whether all validations passed
        confidence: Overall confidence (weighted average)
        uncertainty: Overall uncertainty (max of individual uncertainties)
        validators: Results from individual validators
        needs_review: Whether manual review is recommended
        summary: Human-readable summary
    """

    is_valid: bool
    confidence: float
    uncertainty: float
    validators: list[ValidationResult] = field(default_factory=list)
    needs_review: bool = False
    summary: str = ""
