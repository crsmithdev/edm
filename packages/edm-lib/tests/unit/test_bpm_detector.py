"""Tests for BPM detector module (bpm_detector.py)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from edm.analysis.bpm_detector import (
    ComputedBPM,
    _adjust_bpm_to_edm_range,
    compute_bpm_librosa,
)
from edm.exceptions import AnalysisError

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestComputedBPM:
    """Tests for ComputedBPM dataclass."""

    def test_computed_bpm_creation(self):
        """Test basic ComputedBPM creation."""
        result = ComputedBPM(bpm=128.0, confidence=0.9, method="librosa")
        assert result.bpm == 128.0
        assert result.confidence == 0.9
        assert result.method == "librosa"
        assert result.alternatives == []  # Default empty list

    def test_computed_bpm_with_alternatives(self):
        """Test ComputedBPM with alternatives."""
        result = ComputedBPM(
            bpm=128.0, confidence=0.9, method="beat-this", alternatives=[64.0, 256.0]
        )
        assert result.alternatives == [64.0, 256.0]

    def test_computed_bpm_default_factory_isolation(self):
        """Test that alternatives list is not shared between instances."""
        result1 = ComputedBPM(bpm=128.0, confidence=0.9, method="librosa")
        result2 = ComputedBPM(bpm=140.0, confidence=0.8, method="librosa")

        result1.alternatives.append(64.0)

        assert result1.alternatives == [64.0]
        assert result2.alternatives == []  # Should not be affected


class TestAdjustBpmToEdmRange:
    """Tests for _adjust_bpm_to_edm_range function."""

    def test_bpm_in_range_unchanged(self):
        """Test BPM already in EDM range (120-150) is unchanged."""
        assert _adjust_bpm_to_edm_range(128.0, []) == 128.0
        assert _adjust_bpm_to_edm_range(120.0, []) == 120.0
        assert _adjust_bpm_to_edm_range(150.0, []) == 150.0

    def test_bpm_below_range_adjusted_to_alternative(self):
        """Test BPM below range is adjusted if alternative exists."""
        # 64 BPM -> prefer 128 BPM alternative
        result = _adjust_bpm_to_edm_range(64.0, [128.0])
        assert result == 128.0

    def test_bpm_above_range_adjusted_to_alternative(self):
        """Test BPM above range is adjusted if alternative exists."""
        # 170 BPM -> no alternative in range, stays at 170
        result = _adjust_bpm_to_edm_range(170.0, [85.0, 340.0])
        assert result == 170.0

        # 256 BPM -> prefer 128 BPM alternative
        result = _adjust_bpm_to_edm_range(256.0, [128.0, 512.0])
        assert result == 128.0

    def test_bpm_outside_range_no_alternatives(self):
        """Test BPM outside range with no alternatives stays unchanged."""
        assert _adjust_bpm_to_edm_range(90.0, []) == 90.0
        assert _adjust_bpm_to_edm_range(180.0, []) == 180.0

    def test_bpm_outside_range_alternatives_also_outside(self):
        """Test BPM stays unchanged when alternatives are also outside range."""
        # 60 BPM with alternatives 30 and 120 - should pick 120
        result = _adjust_bpm_to_edm_range(60.0, [30.0, 120.0])
        assert result == 120.0

        # 200 BPM with alternatives 100 and 400 - neither in range
        result = _adjust_bpm_to_edm_range(200.0, [100.0, 400.0])
        assert result == 200.0


class TestComputeBpmLibrosa:
    """Tests for compute_bpm_librosa function."""

    def test_compute_bpm_librosa_mocked_basic(self):
        """Test librosa BPM computation with mocked librosa."""
        with patch("edm.analysis.bpm_detector.librosa") as mock_librosa:
            # Mock beat_track to return reasonable values
            mock_librosa.beat.beat_track.return_value = (
                np.array([128.0]),
                np.array([0, 21, 42, 63, 84]),  # Regular beat frames
            )
            mock_librosa.frames_to_time.return_value = np.array([0.0, 0.469, 0.938, 1.407, 1.876])
            mock_librosa.onset.onset_strength.return_value = np.zeros(100)

            with patch("edm.analysis.bpm_detector.load_audio") as mock_load:
                mock_load.return_value = (np.zeros(22050 * 2), 22050)

                result = compute_bpm_librosa(Path("dummy.mp3"))

                assert isinstance(result, ComputedBPM)
                assert result.method == "librosa"
                assert result.bpm == 128.0
                assert 0.0 <= result.confidence <= 1.0

    def test_compute_bpm_librosa_with_preloaded_audio(self):
        """Test librosa BPM computation with pre-loaded audio (mocked)."""
        with patch("edm.analysis.bpm_detector.librosa") as mock_librosa:
            mock_librosa.beat.beat_track.return_value = (
                np.array([140.0]),
                np.array([0, 18, 36, 54, 72]),
            )
            mock_librosa.frames_to_time.return_value = np.array([0.0, 0.428, 0.857, 1.285, 1.714])
            mock_librosa.onset.onset_strength.return_value = np.zeros(100)

            # Pre-loaded audio
            audio = (np.zeros(22050 * 2), 22050)

            result = compute_bpm_librosa(Path("dummy.mp3"), audio=audio)

            assert isinstance(result, ComputedBPM)
            assert result.method == "librosa"
            assert result.bpm == 140.0

    def test_compute_bpm_librosa_low_beat_confidence(self):
        """Test confidence calculation when few beats detected."""
        with patch("edm.analysis.bpm_detector.librosa") as mock_librosa:
            mock_librosa.beat.beat_track.return_value = (np.array([128.0]), np.array([0]))
            mock_librosa.frames_to_time.return_value = np.array([0.0])

            with patch("edm.analysis.bpm_detector.load_audio") as mock_load:
                mock_load.return_value = (np.zeros(22050), 22050)

                result = compute_bpm_librosa(Path("dummy.mp3"))

                # With only 1 beat, confidence should be 0.5 (default)
                assert result.confidence == 0.5

    def test_compute_bpm_librosa_file_not_found(self):
        """Test librosa BPM computation raises error for nonexistent file."""
        with pytest.raises(AnalysisError, match="BPM computation failed"):
            compute_bpm_librosa(Path("nonexistent_file.mp3"))

    def test_compute_bpm_librosa_generates_alternatives(self):
        """Test that alternatives are generated for tempo multiplicity."""
        with patch("edm.analysis.bpm_detector.librosa") as mock_librosa:
            # BPM of 80 should generate alternative of 160 (double)
            mock_librosa.beat.beat_track.return_value = (
                np.array([80.0]),
                np.array([0, 33, 66, 99]),
            )
            mock_librosa.frames_to_time.return_value = np.array([0.0, 0.75, 1.5, 2.25])
            mock_librosa.onset.onset_strength.return_value = np.zeros(100)

            with patch("edm.analysis.bpm_detector.load_audio") as mock_load:
                mock_load.return_value = (np.zeros(22050 * 3), 22050)

                result = compute_bpm_librosa(Path("dummy.mp3"))

                # Should have alternatives
                assert isinstance(result.alternatives, list)
                # 80 * 0.5 = 40 (in range), 80 * 2 = 160 (in range)
                assert 40.0 in result.alternatives or 160.0 in result.alternatives


class TestComputeBpmLibrosaErrorHandling:
    """Tests for error handling in compute_bpm_librosa."""

    def test_librosa_internal_error_wrapped(self):
        """Test that librosa internal errors are wrapped in AnalysisError."""
        with patch("edm.analysis.bpm_detector.load_audio") as mock_load:
            mock_load.side_effect = RuntimeError("Librosa internal error")

            with pytest.raises(AnalysisError, match="BPM computation failed"):
                compute_bpm_librosa(Path("test.mp3"))

    def test_audio_load_error_propagated(self):
        """Test that audio loading errors are handled."""
        with patch("edm.analysis.bpm_detector.load_audio") as mock_load:
            from edm.exceptions import AudioFileError

            mock_load.side_effect = AudioFileError("Cannot decode audio")

            with pytest.raises(AnalysisError, match="BPM computation failed"):
                compute_bpm_librosa(Path("corrupted.mp3"))


@pytest.mark.slow
class TestBpmDetectorIntegration:
    """Integration tests for BPM detector with various fixtures.

    These tests use actual audio fixtures and librosa, which can be slow
    and may have environment-specific issues (numba/threading).
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize("bpm", [120, 128, 140])
    def test_librosa_on_click_tracks(self, bpm):
        """Test librosa directly on click tracks."""
        audio_file = FIXTURES_DIR / f"click_{bpm}bpm.wav"
        if not audio_file.exists():
            pytest.skip(f"Test fixture not found: {audio_file}")

        result = compute_bpm_librosa(audio_file)

        # 5% tolerance
        tolerance = bpm * 0.05
        assert abs(result.bpm - bpm) <= tolerance, f"Expected {bpm}±{tolerance}, got {result.bpm}"

    @pytest.mark.parametrize("bpm", [120, 128, 140])
    def test_librosa_on_beat_patterns(self, bpm):
        """Test librosa directly on beat patterns."""
        audio_file = FIXTURES_DIR / f"beat_{bpm}bpm.wav"
        if not audio_file.exists():
            pytest.skip(f"Test fixture not found: {audio_file}")

        result = compute_bpm_librosa(audio_file)

        # 5% tolerance
        tolerance = bpm * 0.05
        assert abs(result.bpm - bpm) <= tolerance, f"Expected {bpm}±{tolerance}, got {result.bpm}"
