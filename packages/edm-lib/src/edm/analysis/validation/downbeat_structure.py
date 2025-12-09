"""Downbeat/structure alignment validator."""

import statistics

import numpy as np

from edm.analysis.bpm import BPMResult
from edm.analysis.structure import StructureResult
from edm.analysis.validation.base import BaseValidator
from edm.analysis.validation.results import (
    AlignmentError,
    ErrorPattern,
    ValidationResult,
)


class DownbeatStructureValidator(BaseValidator):
    """Validates alignment between downbeats and structure boundaries."""

    @property
    def name(self) -> str:
        """Validator name."""
        return "downbeat_structure"

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
        tolerance_beats: float = 0.5,
    ) -> ValidationResult:
        """Validate alignment between downbeats and structure.

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
        errors = self._calculate_downbeat_alignment(
            bpm_result,
            structure_result,
            tolerance_beats,
        )

        # Check first downbeat alignment
        first_downbeat_error = self._validate_first_downbeat(
            bpm_result,
            structure_result,
            tolerance_beats,
        )

        if first_downbeat_error:
            errors.insert(0, first_downbeat_error)

        # Detect error pattern
        pattern = self._detect_error_pattern(errors)

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(errors)

        # Determine if valid
        is_valid = pattern == ErrorPattern.NO_ERROR

        # Calculate overall confidence
        if errors:
            mean_abs_error = statistics.mean(abs(e.offset_bars) for e in errors)
            confidence = max(0.0, 1.0 - (mean_abs_error / 2.0))
        else:
            confidence = 1.0

        message = self._format_message(pattern, errors, is_valid)

        return ValidationResult(
            validator_name=self.name,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=uncertainty,
            pattern=pattern,
            errors=errors,
            message=message,
        )

    def _find_nearest_downbeat(
        self,
        time: float,
        bpm: float,
        first_downbeat: float,
    ) -> tuple[float, int]:
        """Find nearest downbeat to a given time.

        Args:
            time: Time to check
            bpm: BPM value
            first_downbeat: Time of first downbeat

        Returns:
            Tuple of (downbeat_time, downbeat_index)
        """
        if time < first_downbeat:
            return first_downbeat, 0

        # Calculate bar duration (4 beats)
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * 4

        # Calculate which downbeat is nearest
        elapsed = time - first_downbeat
        bar_index = round(elapsed / bar_duration)
        downbeat_time = first_downbeat + (bar_index * bar_duration)

        return downbeat_time, int(bar_index)

    def _calculate_downbeat_alignment(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float,
    ) -> list[AlignmentError]:
        """Calculate alignment error for each structure boundary.

        Args:
            bpm_result: BPM result
            structure_result: Structure result
            tolerance_beats: Tolerance in beats

        Returns:
            List of alignment errors
        """
        errors: list[AlignmentError] = []
        bpm = bpm_result.bpm
        first_downbeat = structure_result.downbeat

        # Validate required field
        if first_downbeat is None:
            return errors

        # Skip first section (checked in _validate_first_downbeat)
        for i, section in enumerate(structure_result.sections[1:], start=1):
            # Find nearest downbeat to this boundary
            nearest_downbeat, downbeat_index = self._find_nearest_downbeat(
                section.start_time,
                bpm,
                first_downbeat,
            )

            # Calculate offset in seconds and beats
            offset_seconds = section.start_time - nearest_downbeat
            beat_duration = 60.0 / bpm
            offset_beats = offset_seconds / beat_duration

            # Only count as error if outside tolerance
            if abs(offset_beats) > tolerance_beats:
                error_confidence = min(1.0, abs(offset_beats) / 2.0)
                errors.append(
                    AlignmentError(
                        boundary_index=i,
                        expected_bar=float(downbeat_index + 1),  # Bar numbering starts at 1
                        actual_bar=float(downbeat_index + 1) + (offset_beats / 4.0),
                        offset_bars=offset_beats / 4.0,
                        offset_seconds=offset_seconds,
                        confidence=error_confidence,
                    )
                )

        return errors

    def _validate_first_downbeat(
        self,
        bpm_result: BPMResult,
        structure_result: StructureResult,
        tolerance_beats: float,
    ) -> AlignmentError | None:
        """Check if first downbeat aligns with first structure boundary.

        Args:
            bpm_result: BPM result
            structure_result: Structure result
            tolerance_beats: Tolerance in beats

        Returns:
            AlignmentError if misaligned, None otherwise
        """
        if not structure_result.sections:
            return None

        first_section = structure_result.sections[0]
        first_downbeat = structure_result.downbeat

        # Validate required field
        if first_downbeat is None:
            return None

        offset_seconds = first_section.start_time - first_downbeat

        beat_duration = 60.0 / bpm_result.bpm
        offset_beats = offset_seconds / beat_duration

        # Check if outside tolerance
        if abs(offset_beats) > tolerance_beats:
            error_confidence = min(1.0, abs(offset_beats) / 2.0)
            return AlignmentError(
                boundary_index=0,
                expected_bar=1.0,
                actual_bar=1.0 + (offset_beats / 4.0),
                offset_bars=offset_beats / 4.0,
                offset_seconds=offset_seconds,
                confidence=error_confidence,
            )

        return None

    def _detect_error_pattern(
        self,
        errors: list[AlignmentError],
    ) -> ErrorPattern:
        """Detect error pattern from alignment errors.

        Args:
            errors: List of alignment errors

        Returns:
            Detected error pattern
        """
        if not errors:
            return ErrorPattern.NO_ERROR

        # Single boundary error
        if len(errors) == 1:
            # Check if it's the first downbeat
            if errors[0].boundary_index == 0:
                # Check if it's a ~0.5 bar offset (phase error)
                if 0.4 <= abs(errors[0].offset_bars) <= 0.6:
                    return ErrorPattern.DOWNBEAT_PHASE_ERROR
            return ErrorPattern.SINGLE_BOUNDARY_ERROR

        # Calculate statistics
        offsets = [e.offset_bars for e in errors]
        mean_offset = statistics.mean(offsets)
        std_offset = statistics.stdev(offsets) if len(offsets) > 1 else 0.0

        # Systematic offset (all boundaries off by similar amount)
        if std_offset < 0.1:
            # Check if it's a phase error (~0.5 bars)
            if 0.4 <= abs(mean_offset) <= 0.6:
                return ErrorPattern.DOWNBEAT_PHASE_ERROR
            else:
                return ErrorPattern.BPM_SYSTEMATIC_OFFSET

        # Progressive drift
        if len(errors) >= 3:
            indices = [float(e.boundary_index) for e in errors]
            correlation = np.corrcoef(indices, offsets)[0, 1]
            if abs(correlation) > 0.7:
                return ErrorPattern.PROGRESSIVE_DRIFT

        return ErrorPattern.UNKNOWN

    def _calculate_uncertainty(self, errors: list[AlignmentError]) -> float:
        """Calculate uncertainty from alignment errors.

        Args:
            errors: List of alignment errors

        Returns:
            Uncertainty score [0-1]
        """
        if not errors:
            return 0.0

        # Uncertainty = mean absolute offset, normalized
        mean_abs_offset = statistics.mean(abs(e.offset_bars) for e in errors)
        uncertainty = min(1.0, mean_abs_offset / 1.0)

        return uncertainty

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
            return "All structure boundaries align with downbeats"

        error_count = len(errors)
        pattern_name = pattern.value.replace("_", " ").title()

        if pattern == ErrorPattern.DOWNBEAT_PHASE_ERROR:
            return f"Downbeat phase error detected ({error_count} boundaries ~0.5 bars off)"
        elif pattern == ErrorPattern.SINGLE_BOUNDARY_ERROR:
            if errors and errors[0].boundary_index == 0:
                return "First structure boundary misaligned with first downbeat"
            return f"Single boundary misaligned (index {errors[0].boundary_index})"
        elif pattern == ErrorPattern.BPM_SYSTEMATIC_OFFSET:
            mean_offset = statistics.mean(e.offset_bars for e in errors)
            return f"BPM systematic offset detected ({error_count} boundaries, mean offset: {mean_offset:.2f} bars)"
        else:
            return f"{pattern_name} detected ({error_count} boundaries misaligned)"
