"""Base validator protocol and class."""

from abc import ABC, abstractmethod

from edm.analysis.bpm import BPMResult
from edm.analysis.structure import StructureResult
from edm.analysis.validation.results import ValidationResult


class Validator(ABC):
    """Protocol for cross-validation validators.

    Validators check alignment between different analysis signals
    (BPM, structure, beat grid, downbeat) and report errors.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for identification."""
        ...

    @abstractmethod
    def validate(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float = 0.25,
    ) -> ValidationResult:
        """Validate alignment between signals.

        Args:
            bpm_result: BPM detection result
            structure_result: Structure detection result
            tolerance_beats: Tolerance in beats for alignment (default 1/16 bar)

        Returns:
            Validation result with errors, patterns, and corrections
        """
        ...

    @abstractmethod
    def is_applicable(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
    ) -> bool:
        """Check if validator can run on these inputs.

        Args:
            bpm_result: BPM detection result
            structure_result: Structure detection result

        Returns:
            True if validator can run
        """
        ...


class BaseValidator(Validator):
    """Base implementation with common utilities."""

    def _validate_bpm_source(self, bpm_result: BPMResult) -> bool:
        """Fault detection: check if BPM source is valid.

        Args:
            bpm_result: BPM detection result

        Returns:
            True if BPM is plausible and has sufficient confidence
        """
        if bpm_result.bpm is None:
            return False

        # Check BPM in valid range (40-200 for EDM)
        if not (40 <= bpm_result.bpm <= 200):
            return False

        # Check minimum confidence
        if bpm_result.confidence < 0.3:
            return False

        return True

    def _validate_structure_source(self, structure_result: StructureResult) -> bool:
        """Fault detection: check if structure source is valid.

        Args:
            structure_result: Structure detection result

        Returns:
            True if structure is plausible and has sufficient confidence
        """
        # Need at least 2 sections for meaningful validation
        if len(structure_result.sections) < 2:
            return False

        # Check minimum confidence for all sections
        if not all(s.confidence >= 0.3 for s in structure_result.sections):
            return False

        return True
