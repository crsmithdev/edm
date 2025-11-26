"""Tests for BPM analysis module."""

from pathlib import Path

import pytest

from edm.analysis.bpm import BPMResult, analyze_bpm
from edm.exceptions import AnalysisError

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_bpm_result_dataclass():
    """Test BPMResult dataclass creation."""
    result = BPMResult(bpm=128.0, confidence=0.95, source="computed", method="madmom")
    assert result.bpm == 128.0
    assert result.confidence == 0.95
    assert result.source == "computed"
    assert result.method == "madmom"


def test_analyze_bpm_nonexistent_file():
    """Test BPM analysis with nonexistent file raises appropriate error."""
    with pytest.raises(AnalysisError, match="All BPM lookup strategies failed"):
        analyze_bpm(Path("nonexistent_dummy.mp3"))


@pytest.mark.parametrize(
    "bpm,tolerance",
    [
        (120, 6.0),  # ±5% tolerance
        (125, 6.25),
        (128, 6.4),
        (140, 7.0),
        (150, 7.5),
        (174, 8.7),
    ],
)
def test_analyze_bpm_click_tracks(bpm, tolerance):
    """Test BPM analysis on synthetic click tracks with known BPM.

    Following best practices from librosa/madmom: use synthetic audio
    with known ground truth for unit testing algorithms.
    """
    audio_file = FIXTURES_DIR / f"click_{bpm}bpm.wav"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    # Analyze with forced computation (no metadata lookup)
    result = analyze_bpm(audio_file, offline=True, ignore_metadata=True)

    # Check result structure
    assert isinstance(result, BPMResult)
    assert result.source == "computed"
    assert result.method in ["madmom-dbn", "librosa"]  # Should use one of these
    assert 0.0 <= result.confidence <= 1.0

    # Check BPM accuracy (5% tolerance is standard for tempo detection)
    assert abs(result.bpm - bpm) <= tolerance, (
        f"BPM detection failed: expected {bpm}±{tolerance}, got {result.bpm}"
    )


@pytest.mark.parametrize(
    "bpm,tolerance",
    [
        (120, 6.0),
        (128, 6.4),
        (140, 7.0),
    ],
)
def test_analyze_bpm_beat_patterns(bpm, tolerance):
    """Test BPM analysis on synthetic beat patterns.

    Beat patterns are more complex than click tracks and test the
    algorithm's robustness to realistic drum sounds.
    """
    audio_file = FIXTURES_DIR / f"beat_{bpm}bpm.wav"
    assert audio_file.exists(), f"Test fixture not found: {audio_file}"

    result = analyze_bpm(audio_file, offline=True, ignore_metadata=True)

    assert isinstance(result, BPMResult)
    assert result.source == "computed"
    assert abs(result.bpm - bpm) <= tolerance, (
        f"BPM detection on beat pattern failed: expected {bpm}±{tolerance}, got {result.bpm}"
    )


def test_analyze_bpm_returns_high_confidence():
    """Test that analysis of clear synthetic audio returns high confidence.

    Synthetic click tracks should produce confident results.
    """
    audio_file = FIXTURES_DIR / "click_128bpm.wav"
    result = analyze_bpm(audio_file, offline=True, ignore_metadata=True)

    # Synthetic audio should yield high confidence
    assert result.confidence > 0.7, (
        f"Expected high confidence for synthetic audio, got {result.confidence}"
    )
