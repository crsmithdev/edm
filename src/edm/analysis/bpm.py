"""BPM detection and analysis with cascading lookup strategy."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

from edm.io.audio import AudioData, load_audio

logger = structlog.get_logger(__name__)


@dataclass
class BPMResult:
    """Result of BPM analysis.

    Attributes:
        bpm: Detected BPM value.
        confidence: Confidence score between 0 and 1.
        source: Source of the BPM data ('metadata' or 'computed').
        method: Detection method used (e.g., 'beat-this', 'librosa').
        computation_time: Time spent computing/fetching BPM in seconds.
        alternatives: Alternative BPM candidates (for tempo multiplicity).
    """

    bpm: float
    confidence: float
    source: Literal["metadata", "computed"]
    method: str | None = None
    computation_time: float = 0.0
    alternatives: list[float] = field(default_factory=list)


def analyze_bpm(
    filepath: Path,
    *,
    use_madmom: bool = True,
    use_librosa: bool = False,
    strategy: list[str] | None = None,
    ignore_metadata: bool = False,
    offline: bool = False,
) -> BPMResult:
    """Analyze BPM of an audio track using cascading lookup strategy.

    Default strategy: metadata → computed

    Args:
        filepath: Path to the audio file.
        use_madmom: Use neural network (beat_this) for BPM detection (default: True).
            Parameter name kept for backward compatibility.
        use_librosa: Use librosa for BPM detection (default: False).
        strategy: Custom lookup strategy. If None, uses default based on flags.
        ignore_metadata: Skip metadata lookup (default: False).
        offline: Ignored, kept for backward compatibility. No network calls are made.

    Returns:
        BPM analysis result with confidence score and source.

    Raises:
        AudioFileError: If the audio file cannot be loaded.
        AnalysisError: If BPM detection fails with all methods.

    Examples:
        >>> from pathlib import Path
        >>> result = analyze_bpm(Path("track.mp3"))
        >>> print(f"BPM: {result.bpm:.1f} from {result.source}")
        BPM: 128.0 from metadata

        >>> # Force computation
        >>> result = analyze_bpm(Path("track.mp3"), ignore_metadata=True)
        >>> print(f"BPM: {result.bpm:.1f} (computed using {result.method})")
        BPM: 128.0 (computed using beat-this)
    """
    start_time = time.time()

    # Determine strategy based on flags
    if strategy is None:
        strategy = _build_strategy(ignore_metadata)

    logger.debug("starting bpm analysis", filepath=str(filepath), strategy=strategy)

    # Try each strategy in order
    for source in strategy:
        try:
            if source == "metadata" and not ignore_metadata:
                result = _try_metadata(filepath)
                if result:
                    result.computation_time = time.time() - start_time
                    return result

            elif source == "computed":
                result = _try_compute(filepath, use_madmom, use_librosa)
                if result:
                    result.computation_time = time.time() - start_time
                    return result

        except Exception as e:
            logger.warning("bpm lookup failed", source=source, error=str(e))
            continue

    from edm.exceptions import AnalysisError

    raise AnalysisError(f"All BPM lookup strategies failed for {filepath}")


def _build_strategy(ignore_metadata: bool) -> list[str]:
    """Build lookup strategy based on flags.

    Args:
        ignore_metadata: Skip metadata lookup.

    Returns:
        Ordered list of strategies to try.
    """
    strategy = []

    if not ignore_metadata:
        strategy.append("metadata")

    strategy.append("computed")

    return strategy


def _try_metadata(filepath: Path) -> BPMResult | None:
    """Try to get BPM from file metadata.

    Args:
        filepath: Path to the audio file.

    Returns:
        BPM result if found in metadata, None otherwise.
    """
    logger.debug("trying metadata lookup", filepath=str(filepath))

    try:
        from edm.io.metadata import read_metadata

        metadata = read_metadata(filepath)
        bpm = metadata.get("bpm")

        if bpm and _is_valid_bpm(bpm):
            logger.debug("found bpm in metadata", filepath=str(filepath), bpm=bpm)
            return BPMResult(
                bpm=float(bpm),
                confidence=0.7,  # Metadata confidence
                source="metadata",
                method=None,
            )
        else:
            logger.debug("no valid bpm in metadata", filepath=str(filepath))
            return None

    except Exception as e:
        logger.warning("metadata lookup failed", error=str(e))
        return None


def _try_compute(
    filepath: Path,
    use_madmom: bool,
    use_librosa: bool,
    audio: AudioData | None = None,
) -> BPMResult | None:
    """Try to compute BPM from audio analysis.

    Args:
        filepath: Path to the audio file.
        use_madmom: Prefer neural network (beat_this) for computation.
        use_librosa: Use librosa for computation.
        audio: Pre-loaded audio data as (y, sr) tuple.

    Returns:
        Computed BPM result.
    """
    logger.debug("computing bpm", filepath=str(filepath))

    try:
        from edm.analysis.bpm_detector import compute_bpm

        # Load audio with caching if not already provided
        if audio is None:
            audio = load_audio(filepath)

        result = compute_bpm(filepath, prefer_madmom=use_madmom, audio=audio)

        logger.debug(
            "bpm_computed",
            filepath=str(filepath),
            bpm=round(result.bpm, 1),
            method=result.method,
            confidence=round(result.confidence, 2),
        )
        return BPMResult(
            bpm=result.bpm,
            confidence=result.confidence,
            source="computed",
            method=result.method,
            alternatives=result.alternatives,
        )

    except Exception as e:
        logger.error("bpm computation failed", filepath=str(filepath), error=str(e))
        raise


def _is_valid_bpm(bpm: float) -> bool:
    """Check if BPM value is in valid range.

    Args:
        bpm: BPM value to validate.

    Returns:
        True if BPM is valid, False otherwise.
    """
    return 40.0 <= bpm <= 200.0
