"""Bar/measure calculation utilities for musical time representation.

This module provides utilities to convert between time-based and bar-based
positions in audio tracks. It is designed to work with both constant tempo
(BPM-based) calculations and future beat grid implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import Any  # For future beat_grid parameter

# Type alias for time signature (beats per bar, beat note value)
# Default: (4, 4) = 4/4 time (4 beats per bar, quarter note gets the beat)
TimeSignature = tuple[int, int]


def time_to_bars(
    time_seconds: float,
    bpm: Optional[float],
    time_signature: TimeSignature = (4, 4),
    beat_grid: Optional[Any] = None,
) -> Optional[tuple[int, float]]:
    """Convert time position to bar number and fractional beat.

    Args:
        time_seconds: Time position in seconds.
        bpm: Beats per minute. If None, returns None.
        time_signature: Time signature as (beats_per_bar, beat_note_value).
            Default is (4, 4) for 4/4 time.
        beat_grid: Optional beat grid for precise beat positions. Reserved for
            future implementation. Currently unused.

    Returns:
        Tuple of (bar_number, fractional_beat) or None if BPM unavailable.
        Bar numbers are 1-indexed (bar 1 = first bar). Fractional beat is within the bar (0.0-beats_per_bar).

    Examples:
        >>> time_to_bars(30.5, 128, (4, 4))
        (17, 1.0)  # Bar 17, beat 1
        >>> time_to_bars(0.0, 128, (4, 4))
        (1, 0.0)  # Start of bar 1
        >>> time_to_bars(15.0, None)
        None  # BPM unavailable
    """
    if bpm is None or bpm <= 0:
        return None

    # Future: Use beat_grid if available for precise positions
    # For now: constant tempo calculation

    beats_per_bar, _ = time_signature

    # Convert time to beats
    beats_per_second = bpm / 60.0
    total_beats = time_seconds * beats_per_second

    # Convert beats to bars (1-indexed)
    bar_number = int(total_beats / beats_per_bar) + 1
    fractional_beat = total_beats % beats_per_bar

    return (bar_number, fractional_beat)


def bars_to_time(
    bar: float,
    bpm: Optional[float],
    time_signature: TimeSignature = (4, 4),
    first_downbeat: float = 0.0,
) -> Optional[float]:
    """Convert bar position to time in seconds.

    Args:
        bar: Bar position (1-indexed, can be fractional, e.g., 16.5 = middle of bar 16).
        bpm: Beats per minute. If None, returns None.
        time_signature: Time signature as (beats_per_bar, beat_note_value).
            Default is (4, 4) for 4/4 time.
        first_downbeat: Time in seconds where bar 1 begins. Default 0.0.

    Returns:
        Time in seconds or None if BPM unavailable.

    Examples:
        >>> bars_to_time(17.0, 128, (4, 4))
        30.0  # Bar 17 starts at 30s (16 bars elapsed)
        >>> bars_to_time(1.0, 128, (4, 4))
        0.0  # Bar 1 starts at 0s
        >>> bars_to_time(9.0, 128, (4, 4))
        15.0  # Bar 9 starts at 15s (8 bars elapsed)
        >>> bars_to_time(16.0, None)
        None  # BPM unavailable
        >>> bars_to_time(1.0, 128, (4, 4), first_downbeat=2.0)
        2.0  # Bar 1 starts at 2s when first_downbeat offset is 2.0
    """
    if bpm is None or bpm <= 0:
        return None

    beats_per_bar, _ = time_signature

    # Convert bars to beats (bar 1 = 0 beats elapsed, bar 9 = 8 bars = 32 beats elapsed)
    total_beats = (bar - 1.0) * beats_per_bar

    # Convert beats to time
    beats_per_second = bpm / 60.0
    time_seconds = first_downbeat + (total_beats / beats_per_second)

    return time_seconds


def bar_count_for_range(
    start_time: float,
    end_time: float,
    bpm: Optional[float],
    time_signature: TimeSignature = (4, 4),
    beat_grid: Optional[Any] = None,
) -> Optional[float]:
    """Calculate number of bars spanning a time range.

    Args:
        start_time: Range start in seconds.
        end_time: Range end in seconds.
        bpm: Beats per minute. If None, returns None.
        time_signature: Time signature as (beats_per_bar, beat_note_value).
            Default is (4, 4) for 4/4 time.
        beat_grid: Optional beat grid for precise beat positions. Reserved for
            future implementation. Currently unused.

    Returns:
        Number of bars (can be fractional) or None if BPM unavailable.

    Examples:
        >>> bar_count_for_range(30.0, 60.0, 128, (4, 4))
        16.0  # 30 seconds at 128 BPM in 4/4 = 16 bars
        >>> bar_count_for_range(0.0, 15.0, 128, (4, 4))
        8.0  # 15 seconds = 8 bars
        >>> bar_count_for_range(10.0, 20.0, None)
        None  # BPM unavailable
    """
    if bpm is None or bpm <= 0:
        return None

    # Future: Use beat_grid if available for precise positions
    # For now: constant tempo calculation

    duration = end_time - start_time
    if duration < 0:
        return 0.0

    beats_per_bar, _ = time_signature

    # Convert duration to beats
    beats_per_second = bpm / 60.0
    total_beats = duration * beats_per_second

    # Convert beats to bars
    bar_count = total_beats / beats_per_bar

    return bar_count


def get_section_at_bar(
    bar: float,
    sections: list,
    bpm: Optional[float],
    time_signature: TimeSignature = (4, 4),
) -> Optional[Any]:
    """Get the section containing a specific bar position.

    Args:
        bar: Bar position to query.
        sections: List of sections with start_bar and end_bar attributes.
        bpm: Beats per minute (used for validation).
        time_signature: Time signature for bar calculations.

    Returns:
        The section containing the bar, or None if not found.

    Examples:
        >>> sections = [Section(start_bar=0, end_bar=16), Section(start_bar=16, end_bar=32)]
        >>> get_section_at_bar(20, sections, 128)
        Section(start_bar=16, end_bar=32)
    """
    if bpm is None or bpm <= 0:
        return None

    for section in sections:
        start_bar = getattr(section, "start_bar", None)
        end_bar = getattr(section, "end_bar", None)

        if start_bar is not None and end_bar is not None:
            if start_bar <= bar < end_bar:
                return section

    return None


def check_bar_alignment(
    time_seconds: float,
    bpm: Optional[float],
    time_signature: TimeSignature = (4, 4),
    tolerance: float = 0.5,
) -> Optional[bool]:
    """Check if a time position aligns with a bar boundary.

    Args:
        time_seconds: Time position in seconds.
        bpm: Beats per minute. If None, returns None.
        time_signature: Time signature for bar calculations.
        tolerance: Tolerance in beats for alignment (default ±0.5 beats).

    Returns:
        True if aligned with bar boundary, False if not, None if BPM unavailable.

    Examples:
        >>> check_bar_alignment(30.0, 128, (4, 4), tolerance=0.5)
        True  # Exactly on bar boundary
        >>> check_bar_alignment(30.1, 128, (4, 4), tolerance=0.1)
        False  # Slightly off
    """
    if bpm is None or bpm <= 0:
        return None

    result = time_to_bars(time_seconds, bpm, time_signature)
    if result is None:
        return None

    _, fractional_beat = result

    # Check if fractional beat is close to 0 (start of bar)
    beats_per_bar, _ = time_signature
    return fractional_beat < tolerance or (beats_per_bar - fractional_beat) < tolerance
