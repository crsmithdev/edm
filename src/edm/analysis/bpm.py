"""BPM detection and analysis with cascading lookup strategy."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class BPMResult:
    """Result of BPM analysis.

    Attributes
    ----------
    bpm : float
        Detected BPM value.
    confidence : float
        Confidence score between 0 and 1.
    source : Literal['metadata', 'spotify', 'computed']
        Source of the BPM data.
    method : Optional[str]
        Detection method used (e.g., 'madmom-dbn', 'librosa').
    computation_time : float
        Time spent computing/fetching BPM in seconds.
    alternatives : List[float]
        Alternative BPM candidates (for tempo multiplicity).
    """
    bpm: float
    confidence: float
    source: Literal['metadata', 'spotify', 'computed']
    method: Optional[str] = None
    computation_time: float = 0.0
    alternatives: List[float] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


def analyze_bpm(
    filepath: Path,
    *,
    use_madmom: bool = True,
    use_librosa: bool = False,
    strategy: Optional[List[str]] = None,
    ignore_metadata: bool = False,
    offline: bool = False
) -> BPMResult:
    """Analyze BPM of an audio track using cascading lookup strategy.

    Default strategy: metadata → spotify → computed

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    use_madmom : bool, optional
        Use madmom for BPM detection (default: True).
    use_librosa : bool, optional
        Use librosa for BPM detection (default: False).
    strategy : Optional[List[str]], optional
        Custom lookup strategy. If None, uses default based on flags.
    ignore_metadata : bool, optional
        Skip metadata lookup (default: False).
    offline : bool, optional
        Skip network calls/Spotify API (default: False).

    Returns
    -------
    BPMResult
        BPM analysis result with confidence score and source.

    Raises
    ------
    AudioFileError
        If the audio file cannot be loaded.
    AnalysisError
        If BPM detection fails with all methods.

    Examples
    --------
    >>> from pathlib import Path
    >>> result = analyze_bpm(Path("track.mp3"))
    >>> print(f"BPM: {result.bpm:.1f} from {result.source}")
    BPM: 128.0 from metadata

    >>> # Force computation
    >>> result = analyze_bpm(Path("track.mp3"), offline=True, ignore_metadata=True)
    >>> print(f"BPM: {result.bpm:.1f} (computed using {result.method})")
    BPM: 128.0 (computed using madmom-dbn)
    """
    start_time = time.time()

    # Determine strategy based on flags
    if strategy is None:
        strategy = _build_strategy(ignore_metadata, offline)

    logger.info(f"Analyzing BPM for {filepath} with strategy: {strategy}")

    # Try each strategy in order
    for source in strategy:
        try:
            if source == "metadata" and not ignore_metadata:
                result = _try_metadata(filepath)
                if result:
                    result.computation_time = time.time() - start_time
                    return result

            elif source == "spotify" and not offline:
                result = _try_spotify(filepath)
                if result:
                    result.computation_time = time.time() - start_time
                    return result

            elif source == "computed":
                result = _try_compute(filepath, use_madmom, use_librosa)
                if result:
                    result.computation_time = time.time() - start_time
                    return result

        except Exception as e:
            logger.warning(f"BPM lookup failed for {source}: {e}")
            continue

    from edm.exceptions import AnalysisError
    raise AnalysisError(f"All BPM lookup strategies failed for {filepath}")


def _build_strategy(ignore_metadata: bool, offline: bool) -> List[str]:
    """Build lookup strategy based on flags.

    Parameters
    ----------
    ignore_metadata : bool
        Skip metadata lookup.
    offline : bool
        Skip network calls.

    Returns
    -------
    List[str]
        Ordered list of strategies to try.
    """
    strategy = []

    if not ignore_metadata:
        strategy.append("metadata")

    if not offline:
        strategy.append("spotify")

    strategy.append("computed")

    return strategy


def _try_metadata(filepath: Path) -> Optional[BPMResult]:
    """Try to get BPM from file metadata.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.

    Returns
    -------
    Optional[BPMResult]
        BPM result if found in metadata, None otherwise.
    """
    logger.debug(f"Trying metadata lookup for {filepath}")

    try:
        from edm.io.metadata import read_metadata

        metadata = read_metadata(filepath)
        bpm = metadata.get('bpm')

        if bpm and _is_valid_bpm(bpm):
            logger.info(f"Found BPM {bpm} in file metadata")
            return BPMResult(
                bpm=float(bpm),
                confidence=0.7,  # Metadata confidence
                source='metadata',
                method=None
            )
        else:
            logger.debug("No valid BPM found in metadata")
            return None

    except Exception as e:
        logger.warning(f"Metadata lookup failed: {e}")
        return None


def _try_spotify(filepath: Path) -> Optional[BPMResult]:
    """Try to get BPM from Spotify API.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.

    Returns
    -------
    Optional[BPMResult]
        BPM result if found on Spotify, None otherwise.
    """
    logger.debug(f"Trying Spotify lookup for {filepath}")

    try:
        from edm.external.spotify import SpotifyClient
        from edm.io.metadata import read_metadata

        # Get artist and title from metadata for searching
        metadata = read_metadata(filepath)
        artist = metadata.get('artist')
        title = metadata.get('title')

        if not artist or not title:
            logger.debug("Missing artist or title for Spotify search")
            return None

        # Search Spotify
        client = SpotifyClient()
        track_info = client.search_track(artist, title)

        if track_info and track_info.bpm and _is_valid_bpm(track_info.bpm):
            logger.info(f"Found BPM {track_info.bpm} from Spotify")
            return BPMResult(
                bpm=float(track_info.bpm),
                confidence=0.9,  # Spotify confidence
                source='spotify',
                method=None
            )
        else:
            logger.debug("Track not found on Spotify or no BPM available")
            return None

    except Exception as e:
        logger.warning(f"Spotify lookup failed: {e}")
        return None


def _try_compute(filepath: Path, use_madmom: bool, use_librosa: bool) -> Optional[BPMResult]:
    """Try to compute BPM from audio analysis.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    use_madmom : bool
        Prefer madmom for computation.
    use_librosa : bool
        Use librosa for computation.

    Returns
    -------
    Optional[BPMResult]
        Computed BPM result.
    """
    logger.debug(f"Computing BPM for {filepath}")

    try:
        from edm.analysis.bpm_detector import compute_bpm

        result = compute_bpm(filepath, prefer_madmom=use_madmom)

        logger.info(f"Computed BPM {result.bpm:.1f} using {result.method}")
        return BPMResult(
            bpm=result.bpm,
            confidence=result.confidence,
            source='computed',
            method=result.method,
            alternatives=result.alternatives
        )

    except Exception as e:
        logger.error(f"BPM computation failed: {e}")
        raise


def _is_valid_bpm(bpm: float) -> bool:
    """Check if BPM value is in valid range.

    Parameters
    ----------
    bpm : float
        BPM value to validate.

    Returns
    -------
    bool
        True if BPM is valid, False otherwise.
    """
    return 40.0 <= bpm <= 200.0
