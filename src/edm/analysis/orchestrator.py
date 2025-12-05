"""Analysis orchestration for coordinating multi-stage analysis workflows."""

from pathlib import Path

import structlog

from edm.analysis.bars import TimeSignature
from edm.analysis.beat_detector import BeatGrid, detect_beats
from edm.analysis.bpm import BPMResult, analyze_bpm
from edm.analysis.structure import StructureResult, analyze_structure

logger = structlog.get_logger(__name__)


class AnalysisOrchestrator:
    """Orchestrates multi-stage audio analysis workflows.

    Coordinates BPM, beat, and structure detection to avoid circular dependencies
    while maintaining clean separation of concerns.
    """

    def analyze_full(
        self,
        filepath: Path,
        *,
        detector: str = "auto",
        include_bars: bool = True,
        time_signature: TimeSignature = (4, 4),
        model_path: Path | None = None,
    ) -> tuple[BPMResult | None, BeatGrid | None, StructureResult]:
        """Run full analysis: BPM → beats → structure.

        Args:
            filepath: Path to audio file.
            detector: Structure detection method ('auto', 'msaf', 'energy', 'ml').
            include_bars: Include bar calculations in structure result.
            time_signature: Time signature for bar calculations.
            model_path: Optional path to ML model for structure detection.

        Returns:
            Tuple of (bpm_result, beat_grid, structure_result).
            BPM and beats may be None if detection fails or include_bars=False.

        Raises:
            Exception: If structure detection fails (BPM/beat failures are logged, not raised).
        """
        bpm_result = None
        beat_grid = None

        # Stage 1: BPM detection (if needed for bar calculations)
        if include_bars:
            try:
                logger.debug("analyzing BPM", filepath=str(filepath))
                bpm_result = analyze_bpm(filepath)
                logger.debug("bpm detected", bpm=bpm_result.bpm)
            except Exception as e:
                logger.debug(
                    "bpm analysis failed, continuing without bar calculations", error=str(e)
                )

        # Stage 2: Beat grid detection (if needed and BPM succeeded)
        if include_bars and bpm_result is not None:
            try:
                logger.debug("detecting beat grid", filepath=str(filepath))
                beat_grid = detect_beats(filepath)
                logger.debug("beat grid detected", first_downbeat=beat_grid.first_beat_time)
            except Exception as e:
                logger.debug(
                    "beat detection failed, bar calculations may be incomplete", error=str(e)
                )

        # Stage 3: Structure detection (using BPM from stage 1)
        logger.debug("analyzing structure", detector=detector, filepath=str(filepath))
        structure_result = analyze_structure(
            filepath,
            detector=detector,
            bpm=bpm_result.bpm if bpm_result else None,
            include_bars=include_bars,
            time_signature=time_signature,
            model_path=model_path,
        )

        return bpm_result, beat_grid, structure_result
