"""Unit tests for beat detection functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from edm.analysis.beat_detector import (
    _estimate_downbeats_from_energy,
    detect_beats,
    detect_beats_librosa,
)
from edm.analysis.beat_grid import BeatGrid


class TestEstimateDownbeatsFromEnergy:
    """Tests for downbeat estimation from energy analysis."""

    def test_estimate_downbeats_basic(self) -> None:
        """Test basic downbeat estimation."""
        # 8 beats with higher energy on positions 0, 4 (every 4th beat)
        beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        beat_energies = np.array([1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])

        downbeats = _estimate_downbeats_from_energy(beat_times, beat_energies)

        # Should detect beats 0 and 4 as downbeats
        expected = np.array([0.0, 2.0])
        np.testing.assert_array_almost_equal(downbeats, expected)

    def test_estimate_downbeats_offset_phase(self) -> None:
        """Test downbeat estimation with offset phase."""
        # Higher energy on positions 1, 5 (offset by 1)
        beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        beat_energies = np.array([0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5])

        downbeats = _estimate_downbeats_from_energy(beat_times, beat_energies)

        # Should detect beats 1 and 5 as downbeats
        expected = np.array([0.5, 2.5])
        np.testing.assert_array_almost_equal(downbeats, expected)

    def test_estimate_downbeats_insufficient_beats(self) -> None:
        """Test with fewer beats than beats_per_bar."""
        beat_times = np.array([0.0, 0.5])
        beat_energies = np.array([1.0, 0.5])

        downbeats = _estimate_downbeats_from_energy(beat_times, beat_energies)

        # Should return every 4th beat (just the first one)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(downbeats, expected)

    def test_estimate_downbeats_empty(self) -> None:
        """Test with empty arrays."""
        downbeats = _estimate_downbeats_from_energy(np.array([]), np.array([]))
        assert len(downbeats) == 0


class TestDetectBeatsLibrosa:
    """Tests for librosa-based beat detection."""

    @patch("edm.analysis.beat_detector.load_audio")
    @patch("edm.analysis.beat_detector.librosa")
    def test_detect_beats_librosa_basic(
        self, mock_librosa: MagicMock, mock_load_audio: MagicMock
    ) -> None:
        """Test basic librosa beat detection."""
        # Mock audio loading
        mock_load_audio.return_value = (np.zeros(44100 * 10), 44100)

        # Mock librosa beat tracking
        mock_librosa.beat.beat_track.return_value = (
            120.0,  # tempo
            np.array([0, 22, 44, 66, 88, 110]),  # beat frames
        )
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        mock_librosa.onset.onset_strength.return_value = np.ones(1000)

        from pathlib import Path

        grid = detect_beats_librosa(Path("test.mp3"))

        assert isinstance(grid, BeatGrid)
        assert grid.method == "librosa"
        assert grid.bpm == pytest.approx(120.0)

    @patch("edm.analysis.beat_detector.load_audio")
    @patch("edm.analysis.beat_detector.librosa")
    def test_detect_beats_librosa_with_preloaded_audio(
        self, mock_librosa: MagicMock, mock_load_audio: MagicMock
    ) -> None:
        """Test librosa detection with pre-loaded audio."""
        y = np.zeros(44100 * 10)
        sr = 44100

        mock_librosa.beat.beat_track.return_value = (
            128.0,
            np.array([0, 20, 40, 60]),
        )
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.47, 0.94, 1.41])
        mock_librosa.onset.onset_strength.return_value = np.ones(1000)

        from pathlib import Path

        grid = detect_beats_librosa(Path("test.mp3"), audio=(y, sr))

        # load_audio should not be called when audio is provided
        mock_load_audio.assert_not_called()
        assert grid.method == "librosa"

    @patch("edm.analysis.beat_detector.load_audio")
    @patch("edm.analysis.beat_detector.librosa")
    def test_detect_beats_librosa_insufficient_beats(
        self, mock_librosa: MagicMock, mock_load_audio: MagicMock
    ) -> None:
        """Test error when insufficient beats detected."""
        mock_load_audio.return_value = (np.zeros(44100), 44100)
        mock_librosa.beat.beat_track.return_value = (120.0, np.array([0]))
        mock_librosa.frames_to_time.return_value = np.array([0.0])

        from pathlib import Path

        from edm.exceptions import AnalysisError

        with pytest.raises(AnalysisError, match="Insufficient beats"):
            detect_beats_librosa(Path("test.mp3"))


class TestDetectBeats:
    """Tests for the main detect_beats function."""

    @patch("edm.analysis.beat_detector.detect_beats_beat_this")
    def test_detect_beats_uses_beat_this_first(self, mock_beat_this: MagicMock) -> None:
        """Test that detect_beats tries beat_this first."""
        expected_grid = BeatGrid(
            first_beat_time=0.5,
            bpm=128.0,
            confidence=0.95,
            method="beat-this",
        )
        mock_beat_this.return_value = expected_grid

        from pathlib import Path

        grid = detect_beats(Path("test.mp3"))

        mock_beat_this.assert_called_once()
        assert grid == expected_grid

    @patch("edm.analysis.beat_detector.detect_beats_librosa")
    @patch("edm.analysis.beat_detector.detect_beats_beat_this")
    def test_detect_beats_falls_back_to_librosa(
        self, mock_beat_this: MagicMock, mock_librosa: MagicMock
    ) -> None:
        """Test fallback to librosa when beat_this fails."""
        from edm.exceptions import AnalysisError

        mock_beat_this.side_effect = AnalysisError("beat_this not installed")
        expected_grid = BeatGrid(
            first_beat_time=0.0,
            bpm=120.0,
            confidence=0.8,
            method="librosa",
        )
        mock_librosa.return_value = expected_grid

        from pathlib import Path

        grid = detect_beats(Path("test.mp3"))

        mock_beat_this.assert_called_once()
        mock_librosa.assert_called_once()
        assert grid == expected_grid
        assert grid.method == "librosa"
