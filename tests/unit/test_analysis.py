"""Tests for BPM analysis module."""

from pathlib import Path

import pytest

from edm.analysis.bpm import BPMResult, analyze_bpm
from edm.exceptions import AnalysisError


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

