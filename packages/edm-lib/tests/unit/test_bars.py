"""Tests for bar/measure calculation utilities."""

import pytest
from edm.analysis.bars import (
    bar_count_for_range,
    bars_to_time,
    check_bar_alignment,
    get_section_at_bar,
    time_to_bars,
)


class TestTimeToBars:
    """Tests for time_to_bars function."""

    def test_basic_conversion_4_4(self):
        """Test basic time to bar conversion in 4/4."""
        # At 128 BPM in 4/4: 1 bar = 4 beats = (60/128) * 4 = 1.875 seconds
        result = time_to_bars(0.0, 128, (4, 4))
        assert result == (1, 0.0)  # Bar 1, beat 0 (1-indexed)

        result = time_to_bars(1.875, 128, (4, 4))
        assert result[0] == 2  # Bar 2 starts after 1 bar elapsed
        assert result[1] == pytest.approx(0.0, abs=0.01)

        result = time_to_bars(30.0, 128, (4, 4))
        assert result[0] == 17  # Bar 17 starts after 16 bars elapsed

    def test_fractional_beat_position(self):
        """Test fractional beat positions within a bar."""
        # At 128 BPM: 1 beat = 60/128 = 0.46875 seconds
        beat_duration = 60 / 128
        result = time_to_bars(beat_duration * 2.5, 128, (4, 4))
        assert result[0] == 1  # Still in bar 1 (1-indexed)
        assert result[1] == pytest.approx(2.5, abs=0.01)

    def test_different_time_signatures(self):
        """Test conversions with different time signatures."""
        # 3/4 time: 3 beats per bar
        result = time_to_bars(0.0, 120, (3, 4))
        assert result == (1, 0.0)  # Bar 1 (1-indexed)

        # 6/8 time: 6 beats per bar
        result = time_to_bars(0.0, 120, (6, 8))
        assert result == (1, 0.0)  # Bar 1 (1-indexed)

    def test_none_bpm(self):
        """Test handling of None BPM."""
        result = time_to_bars(30.0, None)
        assert result is None

    def test_negative_bpm(self):
        """Test handling of negative BPM."""
        result = time_to_bars(30.0, -120)
        assert result is None

    def test_zero_bpm(self):
        """Test handling of zero BPM."""
        result = time_to_bars(30.0, 0)
        assert result is None


class TestBarsToTime:
    """Tests for bars_to_time function."""

    def test_basic_conversion(self):
        """Test basic bar to time conversion."""
        # At 128 BPM in 4/4: Bar 17 starts after 16 bars = 30 seconds
        result = bars_to_time(17.0, 128, (4, 4))
        assert result == pytest.approx(30.0, abs=0.01)

        result = bars_to_time(1.0, 128, (4, 4))
        assert result == 0.0  # Bar 1 starts at 0.0

    def test_fractional_bars(self):
        """Test fractional bar positions."""
        result = bars_to_time(16.5, 128, (4, 4))
        assert result is not None
        assert result > bars_to_time(16.0, 128, (4, 4))

    def test_different_bpms(self):
        """Test conversions at different BPMs."""
        # Higher BPM = faster = less time for same bars
        time_120 = bars_to_time(16.0, 120, (4, 4))
        time_140 = bars_to_time(16.0, 140, (4, 4))
        assert time_120 > time_140

    def test_none_bpm(self):
        """Test handling of None BPM."""
        result = bars_to_time(16.0, None)
        assert result is None


class TestBarCountForRange:
    """Tests for bar_count_for_range function."""

    def test_basic_range(self):
        """Test basic bar count calculation."""
        # At 128 BPM: 30 seconds = 16 bars
        result = bar_count_for_range(0.0, 30.0, 128, (4, 4))
        assert result == pytest.approx(16.0, abs=0.01)

    def test_partial_bars(self):
        """Test fractional bar counts."""
        result = bar_count_for_range(0.0, 15.0, 128, (4, 4))
        assert result == pytest.approx(8.0, abs=0.01)

    def test_negative_duration(self):
        """Test handling of negative duration."""
        result = bar_count_for_range(30.0, 10.0, 128, (4, 4))
        assert result == 0.0

    def test_none_bpm(self):
        """Test handling of None BPM."""
        result = bar_count_for_range(0.0, 30.0, None)
        assert result is None


class TestGetSectionAtBar:
    """Tests for get_section_at_bar function."""

    def test_find_section(self):
        """Test finding section at bar position."""
        from dataclasses import dataclass

        @dataclass
        class Section:
            start_bar: float
            end_bar: float
            label: str

        sections = [
            Section(start_bar=1, end_bar=17, label="intro"),  # Bars 1-16
            Section(start_bar=17, end_bar=33, label="buildup"),  # Bars 17-32
            Section(start_bar=33, end_bar=65, label="drop"),  # Bars 33-64
        ]

        result = get_section_at_bar(20, sections, 128)
        assert result is not None
        assert result.label == "buildup"

        result = get_section_at_bar(40, sections, 128)
        assert result is not None
        assert result.label == "drop"

    def test_bar_outside_range(self):
        """Test querying bar outside track range."""
        from dataclasses import dataclass

        @dataclass
        class Section:
            start_bar: float
            end_bar: float

        sections = [Section(start_bar=1, end_bar=17)]  # Bars 1-16

        result = get_section_at_bar(100, sections, 128)
        assert result is None

    def test_none_bpm(self):
        """Test handling of None BPM."""
        result = get_section_at_bar(20, [], None)
        assert result is None


class TestCheckBarAlignment:
    """Tests for check_bar_alignment function."""

    def test_aligned_boundary(self):
        """Test detection of aligned bar boundary."""
        # At 128 BPM: bar boundaries at 0, 1.875, 3.75, ...
        result = check_bar_alignment(0.0, 128, (4, 4))
        assert result is True

        result = check_bar_alignment(1.875, 128, (4, 4), tolerance=0.1)
        assert result is True

    def test_unaligned_position(self):
        """Test detection of unaligned position."""
        result = check_bar_alignment(1.0, 128, (4, 4), tolerance=0.1)
        assert result is False

    def test_tolerance(self):
        """Test tolerance parameter."""
        # Slightly off but within tolerance
        result = check_bar_alignment(1.9, 128, (4, 4), tolerance=0.5)
        assert result is True

        # Outside tolerance
        result = check_bar_alignment(1.5, 128, (4, 4), tolerance=0.1)
        assert result is False

    def test_none_bpm(self):
        """Test handling of None BPM."""
        result = check_bar_alignment(1.875, None)
        assert result is None


class TestRoundTrip:
    """Tests for round-trip conversions between time and bars."""

    def test_time_bars_time(self):
        """Test time -> bars -> time conversion."""
        original_time = 30.5
        bpm = 128
        time_sig = (4, 4)

        bar_result = time_to_bars(original_time, bpm, time_sig)
        assert bar_result is not None

        bar_number, fractional_beat = bar_result
        bar_position = bar_number + (fractional_beat / time_sig[0])

        converted_time = bars_to_time(bar_position, bpm, time_sig)
        assert converted_time == pytest.approx(original_time, abs=0.01)

    def test_bars_time_bars(self):
        """Test bars -> time -> bars conversion."""
        original_bars = 16.5
        bpm = 140
        time_sig = (4, 4)

        time_result = bars_to_time(original_bars, bpm, time_sig)
        assert time_result is not None

        bar_result = time_to_bars(time_result, bpm, time_sig)
        assert bar_result is not None

        bar_number, fractional_beat = bar_result
        converted_bars = bar_number + (fractional_beat / time_sig[0])

        assert converted_bars == pytest.approx(original_bars, abs=0.01)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_high_bpm(self):
        """Test calculations with very high BPM."""
        result = time_to_bars(10.0, 200, (4, 4))
        assert result is not None
        assert result[0] > 0

    def test_very_low_bpm(self):
        """Test calculations with very low BPM."""
        result = time_to_bars(10.0, 60, (4, 4))
        assert result is not None
        assert result[0] >= 1  # 1-indexed, so always >= 1

    def test_zero_time(self):
        """Test conversion of zero time."""
        result = time_to_bars(0.0, 128, (4, 4))
        assert result == (1, 0.0)  # Bar 1 at time 0.0 (1-indexed)

    def test_consistency_across_time_signatures(self):
        """Test that calculations are consistent across time signatures."""
        # Same number of beats should give same time regardless of grouping
        bpm = 120

        # Bar 5 in 4/4 = 4 bars elapsed = 16 beats
        time_4_4 = bars_to_time(5.0, bpm, (4, 4))

        # Bar 9 in 2/4 = 8 bars elapsed = 16 beats
        time_2_4 = bars_to_time(9.0, bpm, (2, 4))

        assert time_4_4 == pytest.approx(time_2_4, abs=0.01)
