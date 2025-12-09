"""Beat/structure alignment validator."""

import statistics

import numpy as np

from edm.analysis.bars import time_to_bars
from edm.analysis.bpm import BPMResult
from edm.analysis.structure import StructureResult
from edm.analysis.validation.base import BaseValidator
from edm.analysis.validation.results import (
    AlignmentError,
    CorrectionProposal,
    ErrorPattern,
    ValidationResult,
)


class BeatStructureValidator(BaseValidator):
    """Validates alignment between beat grid and structure boundaries."""

    @property
    def name(self) -> str:
        """Validator name."""
        return "beat_structure"

    def is_applicable(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
    ) -> bool:
        """Check if validator can run.

        Args:
            bpm_result: BPM detection result
            structure_result: Structure detection result

        Returns:
            True if both sources are valid
        """
        return self._validate_bpm_source(bpm_result) and self._validate_structure_source(
            structure_result
        )

    def validate(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float = 0.25,
    ) -> ValidationResult:
        """Validate alignment between beat grid and structure.

        Args:
            bpm_result: BPM detection result
            structure_result: Structure detection result
            tolerance_beats: Tolerance in beats for alignment

        Returns:
            Validation result
        """
        if not self.is_applicable(bpm_result, structure_result):
            return ValidationResult(
                validator_name=self.name,
                is_valid=False,
                confidence=0.0,
                uncertainty=1.0,
                pattern=ErrorPattern.UNKNOWN,
                message="Insufficient data quality for validation",
            )

        # Calculate alignment errors for each boundary
        errors = self._calculate_alignment_errors(
            bpm_result,
            structure_result,
            tolerance_beats,
        )

        # Detect error pattern from distribution
        pattern = self._detect_error_pattern(errors, tolerance_beats)

        # Calculate uncertainty (signal disagreement)
        uncertainty = self._calculate_uncertainty(errors)

        # Determine if valid
        is_valid = pattern == ErrorPattern.NO_ERROR

        # Calculate overall confidence (inverse of mean absolute error)
        if errors:
            mean_abs_error = statistics.mean(abs(e.offset_bars) for e in errors)
            confidence = max(0.0, 1.0 - (mean_abs_error / 2.0))  # Normalize to [0,1]
        else:
            confidence = 1.0

        # Generate correction proposals for invalid cases
        corrections = []
        # Calculate average structure confidence from sections
        structure_confidence = (
            statistics.mean(s.confidence for s in structure_result.sections)
            if structure_result.sections
            else 0.0
        )
        if not is_valid and bpm_result.confidence > structure_confidence:
            corrections = self._propose_structure_corrections(errors, bpm_result.confidence)

        message = self._format_message(pattern, errors, is_valid)

        return ValidationResult(
            validator_name=self.name,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=uncertainty,
            pattern=pattern,
            errors=errors,
            corrections=corrections,
            message=message,
        )

    def _calculate_alignment_errors(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float,
    ) -> list[AlignmentError]:
        """Calculate alignment error for each structure boundary.

        Args:
            bpm_result: BPM result with BPM and downbeat
            structure_result: Structure result with sections
            tolerance_beats: Tolerance in beats

        Returns:
            List of alignment errors
        """
        errors: list[AlignmentError] = []
        bpm = bpm_result.bpm
        downbeat = structure_result.downbeat
        time_signature = structure_result.time_signature

        # Validate required fields
        if downbeat is None or time_signature is None:
            return errors

        for i, section in enumerate(structure_result.sections):
            # Get expected bar from structure
            expected_bar_value = section.start_bar

            if expected_bar_value is None:
                # Calculate bar from time
                bar_result = time_to_bars(section.start_time, bpm, time_signature, downbeat)
                if bar_result is None:
                    continue
                bar_num, _beat_fraction = bar_result
                expected_bar_value = float(bar_num)

            # Calculate actual bar position from beat grid
            # Bar calculation: bar = floor((time - downbeat) / (60/bpm * 4)) + 1
            if section.start_time < downbeat:
                actual_bar = 1.0
            else:
                elapsed = section.start_time - downbeat
                beat_duration = 60.0 / bpm
                bar_duration = beat_duration * 4
                actual_bar = (elapsed / bar_duration) + 1.0

            # Calculate offsets
            offset_bars = expected_bar_value - actual_bar
            offset_seconds = offset_bars * (60.0 / bpm) * 4

            # Only count as error if outside tolerance
            if abs(offset_bars * 4) > tolerance_beats:  # Convert bars to beats
                error_confidence = min(
                    1.0, abs(offset_bars) / 0.5
                )  # Higher offset = higher confidence it's an error
                errors.append(
                    AlignmentError(
                        boundary_index=i,
                        expected_bar=expected_bar_value,
                        actual_bar=actual_bar,
                        offset_bars=offset_bars,
                        offset_seconds=offset_seconds,
                        confidence=error_confidence,
                    )
                )

        return errors

    def _detect_error_pattern(
        self,
        errors: list[AlignmentError],
        tolerance_beats: float,
    ) -> ErrorPattern:
        """Detect error pattern from alignment errors.

        Args:
            errors: List of alignment errors
            tolerance_beats: Tolerance in beats

        Returns:
            Detected error pattern
        """
        if not errors:
            return ErrorPattern.NO_ERROR

        offsets = [e.offset_bars for e in errors]

        # Check for converter bug (all expected_bar values at 1, which is wrong)
        # Skip first boundary since it legitimately starts at bar 1
        if len(errors) >= 2 and all(abs(e.expected_bar - 1.0) < 0.01 for e in errors[1:]):
            return ErrorPattern.CONVERTER_BUG

        # Single boundary error
        if len(errors) == 1:
            return ErrorPattern.SINGLE_BOUNDARY_ERROR

        # Calculate statistics
        mean_offset = statistics.mean(offsets)
        std_offset = statistics.stdev(offsets) if len(offsets) > 1 else 0.0

        # Systematic offset (low std dev, consistent offset)
        if std_offset < 0.1 and abs(mean_offset) > 0.1:
            # Check if it's a phase error (~0.5 bars)
            if 0.4 <= abs(mean_offset) <= 0.6:
                return ErrorPattern.DOWNBEAT_PHASE_ERROR
            # Check if it's time signature mismatch (~0.25 bars)
            elif 0.2 <= abs(mean_offset) <= 0.3:
                return ErrorPattern.TIME_SIGNATURE_MISMATCH
            else:
                return ErrorPattern.BPM_SYSTEMATIC_OFFSET

        # Progressive drift (error increases over time)
        if len(errors) >= 3:
            # Check if errors correlate with time
            indices = [float(e.boundary_index) for e in errors]
            correlation = np.corrcoef(indices, offsets)[0, 1]
            if abs(correlation) > 0.7:
                return ErrorPattern.PROGRESSIVE_DRIFT

        return ErrorPattern.UNKNOWN

    def _calculate_uncertainty(self, errors: list[AlignmentError]) -> float:
        """Calculate uncertainty from alignment errors.

        Uncertainty measures signal disagreement, separate from confidence.

        Args:
            errors: List of alignment errors

        Returns:
            Uncertainty score [0-1]
        """
        if not errors:
            return 0.0

        # Uncertainty = mean absolute offset, normalized
        mean_abs_offset = statistics.mean(abs(e.offset_bars) for e in errors)
        uncertainty = min(1.0, mean_abs_offset / 1.0)  # Clamp to [0, 1]

        return uncertainty

    def _propose_structure_corrections(
        self,
        errors: list[AlignmentError],
        bpm_confidence: float,
    ) -> list[CorrectionProposal]:
        """Propose structure boundary corrections.

        Args:
            errors: List of alignment errors
            bpm_confidence: BPM confidence score

        Returns:
            List of correction proposals
        """
        corrections = []

        for error in errors:
            correction = CorrectionProposal(
                action="quantize_structure",
                target="structure",
                old_value=f"bar {error.expected_bar:.2f}",
                new_value=f"bar {error.actual_bar:.2f}",
                confidence=bpm_confidence * 0.8,  # Slightly reduce confidence for correction
                reason=f"Quantize boundary {error.boundary_index} to nearest bar (BPM confidence: {bpm_confidence:.2f})",
            )
            corrections.append(correction)

        return corrections

    def _format_message(
        self,
        pattern: ErrorPattern,
        errors: list[AlignmentError],
        is_valid: bool,
    ) -> str:
        """Format human-readable validation message.

        Args:
            pattern: Detected error pattern
            errors: List of alignment errors
            is_valid: Whether validation passed

        Returns:
            Human-readable message
        """
        if is_valid:
            return "All structure boundaries align with beat grid"

        error_count = len(errors)
        pattern_name = pattern.value.replace("_", " ").title()

        if pattern == ErrorPattern.CONVERTER_BUG:
            return f"Converter bug detected: all bars at 1 ({error_count} boundaries affected)"
        elif pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR:
            return f"Single boundary misaligned (index {errors[0].boundary_index})"
        elif pattern == ErrorPattern.DOWNBEAT_PHASE_ERROR:
            return f"Downbeat phase error detected ({error_count} boundaries ~0.5 bars off)"
        elif pattern == ErrorPattern.BPM_SYSTEMATIC_OFFSET:
            mean_offset = statistics.mean(e.offset_bars for e in errors)
            return f"BPM systematic offset detected ({error_count} boundaries, mean offset: {mean_offset:.2f} bars)"
        else:
            return f"{pattern_name} detected ({error_count} boundaries misaligned)"
