"""Cross-validation framework for EDM analysis.

Validates alignment between BPM detection, beat grid, and structure detection.
Detects systematic errors and provides confidence-weighted arbitration.
"""

from edm.analysis.validation.base import Validator
from edm.analysis.validation.orchestrator import ValidationOrchestrator, validate_analysis
from edm.analysis.validation.results import (
    AlignmentError,
    CorrectionProposal,
    CrossValidationResult,
    ErrorPattern,
    ValidationResult,
)

__all__ = [
    "Validator",
    "ValidationOrchestrator",
    "validate_analysis",
    "ValidationResult",
    "CrossValidationResult",
    "AlignmentError",
    "ErrorPattern",
    "CorrectionProposal",
]
