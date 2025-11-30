"""Unit tests for BeatGrid class."""

import numpy as np
import pytest

from edm.analysis.beat_grid import BeatGrid


class TestBeatGridBasic:
    """Tests for BeatGrid basic functionality."""

    def test_create_beat_grid(self) -> None:
        """Test creating a BeatGrid with all parameters."""
        grid = BeatGrid(
            first_beat_time=0.5,
            bpm=128.0,
            time_signature=(4, 4),
            confidence=0.95,
            method="beat-this",
        )
        assert grid.first_beat_time == 0.5
        assert grid.bpm == 128.0
        assert grid.time_signature == (4, 4)
        assert grid.confidence == 0.95
        assert grid.method == "beat-this"

    def test_default_values(self) -> None:
        """Test BeatGrid default values."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        assert grid.time_signature == (4, 4)
        assert grid.confidence == 0.0
        assert grid.method == "beat-this"

    def test_first_downbeat_property(self) -> None:
        """Test first_downbeat is alias for first_beat_time."""
        grid = BeatGrid(first_beat_time=1.23, bpm=128.0)
        assert grid.first_downbeat == 1.23
        assert grid.first_downbeat == grid.first_beat_time

    def test_beats_per_bar_property(self) -> None:
        """Test beats_per_bar extracts from time signature."""
        grid_4_4 = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(4, 4))
        assert grid_4_4.beats_per_bar == 4

        grid_3_4 = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(3, 4))
        assert grid_3_4.beats_per_bar == 3

    def test_beat_duration_property(self) -> None:
        """Test beat_duration calculation."""
        grid_120 = BeatGrid(first_beat_time=0.0, bpm=120.0)
        assert grid_120.beat_duration == 0.5  # 60/120 = 0.5s per beat

        grid_128 = BeatGrid(first_beat_time=0.0, bpm=128.0)
        assert grid_128.beat_duration == pytest.approx(0.46875)  # 60/128


class TestBeatToTime:
    """Tests for beat_to_time conversion."""

    def test_beat_to_time_at_origin(self) -> None:
        """Test beat 0 returns first_beat_time."""
        grid = BeatGrid(first_beat_time=0.5, bpm=120.0)
        assert grid.beat_to_time(0) == 0.5

    def test_beat_to_time_sequential(self) -> None:
        """Test sequential beats at correct intervals."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)  # 0.5s per beat
        assert grid.beat_to_time(0) == 0.0
        assert grid.beat_to_time(1) == 0.5
        assert grid.beat_to_time(2) == 1.0
        assert grid.beat_to_time(4) == 2.0

    def test_beat_to_time_with_offset(self) -> None:
        """Test beat_to_time with non-zero first_beat_time."""
        grid = BeatGrid(first_beat_time=2.0, bpm=120.0)
        assert grid.beat_to_time(0) == 2.0
        assert grid.beat_to_time(1) == 2.5
        assert grid.beat_to_time(4) == 4.0


class TestTimeToBeat:
    """Tests for time_to_beat conversion."""

    def test_time_to_beat_at_origin(self) -> None:
        """Test first_beat_time returns beat 0."""
        grid = BeatGrid(first_beat_time=0.5, bpm=120.0)
        assert grid.time_to_beat(0.5) == 0.0

    def test_time_to_beat_sequential(self) -> None:
        """Test sequential times at correct beat indices."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        assert grid.time_to_beat(0.0) == 0.0
        assert grid.time_to_beat(0.5) == 1.0
        assert grid.time_to_beat(1.0) == 2.0
        assert grid.time_to_beat(2.0) == 4.0

    def test_time_to_beat_fractional(self) -> None:
        """Test fractional beat indices."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        assert grid.time_to_beat(0.25) == 0.5
        assert grid.time_to_beat(0.75) == 1.5

    def test_time_to_beat_negative(self) -> None:
        """Test time before first_beat_time returns negative."""
        grid = BeatGrid(first_beat_time=2.0, bpm=120.0)
        assert grid.time_to_beat(1.5) == -1.0
        assert grid.time_to_beat(0.0) == -4.0


class TestToBeatTimes:
    """Tests for to_beat_times timestamp generation."""

    def test_to_beat_times_basic(self) -> None:
        """Test generating beat timestamps."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        beats = grid.to_beat_times(2.0)
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(beats, expected)

    def test_to_beat_times_with_offset(self) -> None:
        """Test beat times with non-zero first_beat_time."""
        grid = BeatGrid(first_beat_time=0.5, bpm=120.0)
        beats = grid.to_beat_times(2.0)
        expected = np.array([0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(beats, expected)

    def test_to_beat_times_duration_before_first_beat(self) -> None:
        """Test duration before first_beat_time returns empty."""
        grid = BeatGrid(first_beat_time=5.0, bpm=120.0)
        beats = grid.to_beat_times(2.0)
        assert len(beats) == 0

    def test_to_beat_times_zero_bpm(self) -> None:
        """Test zero BPM returns empty array."""
        grid = BeatGrid(first_beat_time=0.0, bpm=0.0)
        beats = grid.to_beat_times(10.0)
        assert len(beats) == 0


class TestToDownbeatTimes:
    """Tests for to_downbeat_times generation."""

    def test_to_downbeat_times_4_4(self) -> None:
        """Test downbeat generation in 4/4 time."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(4, 4))
        downbeats = grid.to_downbeat_times(4.0)
        # 4/4 at 120 BPM = 2s per bar
        expected = np.array([0.0, 2.0, 4.0])
        np.testing.assert_array_almost_equal(downbeats, expected)

    def test_to_downbeat_times_3_4(self) -> None:
        """Test downbeat generation in 3/4 time."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(3, 4))
        downbeats = grid.to_downbeat_times(3.0)
        # 3/4 at 120 BPM = 1.5s per bar
        expected = np.array([0.0, 1.5, 3.0])
        np.testing.assert_array_almost_equal(downbeats, expected)

    def test_to_downbeat_times_with_offset(self) -> None:
        """Test downbeat times with non-zero first_beat_time."""
        grid = BeatGrid(first_beat_time=1.0, bpm=120.0, time_signature=(4, 4))
        downbeats = grid.to_downbeat_times(5.0)
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(downbeats, expected)


class TestTimeToBar:
    """Tests for time_to_bar conversion."""

    def test_time_to_bar_at_origin(self) -> None:
        """Test first_beat_time is bar 1, beat 0."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        bar, beat = grid.time_to_bar(0.0)
        assert bar == 1
        assert beat == pytest.approx(0.0)

    def test_time_to_bar_sequential(self) -> None:
        """Test sequential bars at correct positions."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(4, 4))
        # Bar 1 at 0s, Bar 2 at 2s (4 beats * 0.5s)
        bar1, beat1 = grid.time_to_bar(0.0)
        assert bar1 == 1
        assert beat1 == pytest.approx(0.0)

        bar2, beat2 = grid.time_to_bar(2.0)
        assert bar2 == 2
        assert beat2 == pytest.approx(0.0)

    def test_time_to_bar_fractional_beat(self) -> None:
        """Test fractional beat positions within a bar."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        bar, beat = grid.time_to_bar(0.5)  # Beat 1 of bar 1
        assert bar == 1
        assert beat == pytest.approx(1.0)

        bar, beat = grid.time_to_bar(1.5)  # Beat 3 of bar 1
        assert bar == 1
        assert beat == pytest.approx(3.0)

    def test_time_to_bar_with_offset(self) -> None:
        """Test bar calculation with non-zero first_beat_time."""
        grid = BeatGrid(first_beat_time=2.0, bpm=120.0)
        bar, beat = grid.time_to_bar(2.0)
        assert bar == 1
        assert beat == pytest.approx(0.0)

        bar, beat = grid.time_to_bar(4.0)
        assert bar == 2
        assert beat == pytest.approx(0.0)


class TestBarToTime:
    """Tests for bar_to_time conversion."""

    def test_bar_to_time_bar_1(self) -> None:
        """Test bar 1 returns first_beat_time."""
        grid = BeatGrid(first_beat_time=0.5, bpm=120.0)
        assert grid.bar_to_time(1) == pytest.approx(0.5)

    def test_bar_to_time_sequential(self) -> None:
        """Test sequential bars at correct times."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0, time_signature=(4, 4))
        assert grid.bar_to_time(1) == pytest.approx(0.0)
        assert grid.bar_to_time(2) == pytest.approx(2.0)
        assert grid.bar_to_time(3) == pytest.approx(4.0)

    def test_bar_to_time_with_beat(self) -> None:
        """Test bar_to_time with beat offset."""
        grid = BeatGrid(first_beat_time=0.0, bpm=120.0)
        assert grid.bar_to_time(1, beat=0.0) == pytest.approx(0.0)
        assert grid.bar_to_time(1, beat=1.0) == pytest.approx(0.5)
        assert grid.bar_to_time(1, beat=2.0) == pytest.approx(1.0)


class TestFromTimestamps:
    """Tests for BeatGrid.from_timestamps class method."""

    def test_from_timestamps_basic(self) -> None:
        """Test creating BeatGrid from beat timestamps."""
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # 120 BPM
        grid = BeatGrid.from_timestamps(beats)
        assert grid.first_beat_time == 0.0
        assert grid.bpm == pytest.approx(120.0)
        assert grid.time_signature == (4, 4)  # Default

    def test_from_timestamps_with_downbeats(self) -> None:
        """Test creating BeatGrid with downbeat timestamps."""
        beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        downbeats = np.array([0.5, 2.5, 4.5])  # Every 4 beats
        grid = BeatGrid.from_timestamps(beats, downbeats)
        assert grid.first_beat_time == 0.5
        assert grid.bpm == pytest.approx(120.0)
        assert grid.beats_per_bar == 4

    def test_from_timestamps_3_4_time(self) -> None:
        """Test inferring 3/4 time signature."""
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        downbeats = np.array([0.0, 1.5, 3.0])  # Every 3 beats
        grid = BeatGrid.from_timestamps(beats, downbeats)
        assert grid.beats_per_bar == 3

    def test_from_timestamps_preserves_confidence(self) -> None:
        """Test confidence and method are preserved."""
        beats = np.array([0.0, 0.5, 1.0])
        grid = BeatGrid.from_timestamps(beats, confidence=0.9, method="librosa")
        assert grid.confidence == 0.9
        assert grid.method == "librosa"

    def test_from_timestamps_insufficient_beats(self) -> None:
        """Test error with fewer than 2 beats."""
        with pytest.raises(ValueError, match="at least 2 beats"):
            BeatGrid.from_timestamps(np.array([0.0]))

    def test_from_timestamps_empty_beats(self) -> None:
        """Test error with empty beats array."""
        with pytest.raises(ValueError, match="at least 2 beats"):
            BeatGrid.from_timestamps(np.array([]))

    def test_from_timestamps_robust_to_outliers(self) -> None:
        """Test BPM calculation uses median for outlier robustness."""
        # One outlier interval won't affect median
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5])  # 2.5 is outlier
        grid = BeatGrid.from_timestamps(beats)
        assert grid.bpm == pytest.approx(120.0)  # Median interval is 0.5
