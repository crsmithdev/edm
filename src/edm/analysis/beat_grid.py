"""Beat grid representation for index-based beat position storage.

This module provides a compact, deterministic representation of beat positions
using an index-based approach (following the Mixxx pattern) rather than storing
timestamp arrays. Beat positions are generated on demand from grid parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Type alias for time signature (beats per bar, beat note value)
TimeSignature = tuple[int, int]


@dataclass
class BeatGrid:
    """Index-based representation of beat positions.

    Stores minimal state (anchor point, BPM, time signature) and generates
    beat/downbeat timestamps on demand. This is more compact than storing
    arrays and enables deterministic regeneration.

    Attributes:
        first_beat_time: Time in seconds of the first downbeat (bar 1, beat 1).
        bpm: Tempo in beats per minute.
        time_signature: Time signature as (beats_per_bar, beat_note_value).
            Default is (4, 4) for 4/4 time.
        confidence: Detection confidence score (0.0 to 1.0).
        method: Detection method used ("beat-this" or "librosa").
    """

    first_beat_time: float
    bpm: float
    time_signature: TimeSignature = (4, 4)
    confidence: float = 0.0
    method: Literal["beat-this", "librosa"] = "beat-this"

    @property
    def first_downbeat(self) -> float:
        """Return the time of the first downbeat (alias for first_beat_time)."""
        return self.first_beat_time

    @property
    def beats_per_bar(self) -> int:
        """Return the number of beats per bar from time signature."""
        return self.time_signature[0]

    @property
    def beat_duration(self) -> float:
        """Return the duration of one beat in seconds."""
        return 60.0 / self.bpm

    def beat_to_time(self, beat_index: int) -> float:
        """Convert a beat index to time in seconds.

        Args:
            beat_index: Zero-based beat index (0 = first beat at first_beat_time).

        Returns:
            Time in seconds for the given beat index.
        """
        return self.first_beat_time + (beat_index * self.beat_duration)

    def time_to_beat(self, time: float) -> float:
        """Convert a time in seconds to a beat index (float).

        Args:
            time: Time in seconds.

        Returns:
            Beat index as float (can be fractional for positions between beats).
            Negative values indicate time before first_beat_time.
        """
        return (time - self.first_beat_time) / self.beat_duration

    def to_beat_times(self, duration: float) -> np.ndarray:
        """Generate array of beat timestamps up to the given duration.

        Args:
            duration: Track duration in seconds.

        Returns:
            Array of beat timestamps in seconds.
        """
        if self.bpm <= 0:
            return np.array([])

        # Calculate number of beats from first_beat_time to duration
        beats_after_first = int((duration - self.first_beat_time) / self.beat_duration)
        if beats_after_first < 0:
            return np.array([])

        # Generate beat times
        beat_indices = np.arange(beats_after_first + 1)
        return self.first_beat_time + (beat_indices * self.beat_duration)

    def to_downbeat_times(self, duration: float) -> np.ndarray:
        """Generate array of downbeat timestamps up to the given duration.

        Downbeats are the first beat of each bar (every Nth beat based on
        time signature).

        Args:
            duration: Track duration in seconds.

        Returns:
            Array of downbeat timestamps in seconds.
        """
        if self.bpm <= 0:
            return np.array([])

        bar_duration = self.beats_per_bar * self.beat_duration
        bars_after_first = int((duration - self.first_beat_time) / bar_duration)
        if bars_after_first < 0:
            return np.array([])

        bar_indices = np.arange(bars_after_first + 1)
        return self.first_beat_time + (bar_indices * bar_duration)

    def time_to_bar(self, time: float) -> tuple[int, float]:
        """Convert a time in seconds to bar number and fractional beat.

        Args:
            time: Time in seconds.

        Returns:
            Tuple of (bar_number, fractional_beat).
            Bar numbers are 1-indexed (bar 1 = first bar).
            Fractional beat is position within the bar (0.0 to beats_per_bar).
            For times before first_beat_time, returns negative bar numbers.
        """
        beat_index = self.time_to_beat(time)
        bar_index = beat_index / self.beats_per_bar
        bar_number = int(bar_index) + 1 if bar_index >= 0 else int(bar_index)
        fractional_beat = beat_index % self.beats_per_bar
        return (bar_number, fractional_beat)

    def bar_to_time(self, bar: int, beat: float = 0.0) -> float:
        """Convert a bar number and beat to time in seconds.

        Args:
            bar: Bar number (1-indexed, bar 1 = first bar).
            beat: Beat within the bar (0.0 to beats_per_bar - 1).

        Returns:
            Time in seconds.
        """
        # Convert 1-indexed bar to 0-indexed
        bar_index = bar - 1
        total_beats = (bar_index * self.beats_per_bar) + beat
        return self.beat_to_time(int(total_beats)) + (beat % 1) * self.beat_duration

    @classmethod
    def from_timestamps(
        cls,
        beats: np.ndarray,
        downbeats: np.ndarray | None = None,
        confidence: float = 0.0,
        method: Literal["beat-this", "librosa"] = "beat-this",
    ) -> BeatGrid:
        """Create a BeatGrid from raw timestamp arrays.

        Derives grid parameters (first_beat_time, bpm, time_signature) from
        the provided beat and downbeat timestamps.

        Args:
            beats: Array of beat timestamps in seconds.
            downbeats: Optional array of downbeat timestamps in seconds.
                If provided, first_beat_time is set to the first downbeat
                and time signature is inferred from beat/downbeat ratio.
            confidence: Detection confidence score.
            method: Detection method used.

        Returns:
            BeatGrid with parameters derived from timestamps.

        Raises:
            ValueError: If beats array is empty or has fewer than 2 elements.
        """
        if len(beats) < 2:
            raise ValueError("Need at least 2 beats to create BeatGrid")

        # Calculate BPM from median beat interval for robustness
        beat_intervals = np.diff(beats)
        median_interval = float(np.median(beat_intervals))
        bpm = 60.0 / median_interval

        # Determine first_beat_time and time signature
        if downbeats is not None and len(downbeats) > 0:
            first_beat_time = float(downbeats[0])

            # Infer beats per bar from downbeat spacing
            if len(downbeats) >= 2:
                downbeat_intervals = np.diff(downbeats)
                median_downbeat_interval = float(np.median(downbeat_intervals))
                beats_per_bar = round(median_downbeat_interval / median_interval)
                beats_per_bar = max(1, min(beats_per_bar, 12))  # Clamp to reasonable range
            else:
                beats_per_bar = 4  # Default
        else:
            first_beat_time = float(beats[0])
            beats_per_bar = 4  # Default 4/4

        return cls(
            first_beat_time=first_beat_time,
            bpm=bpm,
            time_signature=(beats_per_bar, 4),
            confidence=confidence,
            method=method,
        )
