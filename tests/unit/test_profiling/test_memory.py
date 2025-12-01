"""Tests for memory profiler."""

import pytest

from edm.profiling.base import ProfileResult, profiling_context
from edm.profiling.memory import MemoryProfiler, compare_memory


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_basic_profiling(self):
        """Test basic memory profiling."""
        profiler = MemoryProfiler()
        profiler.start()

        # Allocate some memory
        data = [i * i for i in range(10000)]
        assert len(data) == 10000

        result = profiler.stop()

        assert result is not None
        assert result.profile_type == "memory"
        assert result.wall_time > 0
        assert result.peak_memory_mb >= 0

    def test_context_manager(self):
        """Test profiler as context manager."""
        with MemoryProfiler() as profiler:
            # Allocate some memory
            data = list(range(10000))
            assert len(data) == 10000

        assert profiler.result is not None
        assert profiler.result.peak_memory_mb >= 0
        assert not profiler.is_running

    def test_is_running_state(self):
        """Test is_running property."""
        profiler = MemoryProfiler()
        assert not profiler.is_running

        profiler.start()
        assert profiler.is_running

        profiler.stop()
        assert not profiler.is_running

    def test_stop_without_start_raises(self):
        """Test stopping without starting raises error."""
        profiler = MemoryProfiler()
        with pytest.raises(RuntimeError, match="not running"):
            profiler.stop()

    def test_take_snapshot(self):
        """Test taking memory snapshots."""
        with MemoryProfiler() as profiler:
            # Allocate some memory
            data1 = list(range(5000))

            # Take snapshot
            snapshot1 = profiler.take_snapshot()

            # Allocate more
            data2 = list(range(5000))

            # Take another snapshot
            snapshot2 = profiler.take_snapshot()

            # Keep references to avoid GC
            assert len(data1) == 5000
            assert len(data2) == 5000

        # Should have two snapshots
        assert len(profiler.snapshots) == 2
        assert snapshot1.timestamp < snapshot2.timestamp

    def test_snapshot_without_start_raises(self):
        """Test taking snapshot without starting raises error."""
        profiler = MemoryProfiler()
        with pytest.raises(RuntimeError, match="not running"):
            profiler.take_snapshot()

    def test_top_allocations(self):
        """Test getting top allocations."""
        with MemoryProfiler(top_n=5) as profiler:
            # Allocate memory
            data = [list(range(1000)) for _ in range(100)]
            assert len(data) == 100

        allocations = profiler.top_allocations()
        assert isinstance(allocations, list)

        # If we got allocations, check structure
        if allocations:
            alloc = allocations[0]
            assert "file" in alloc
            assert "line" in alloc
            assert "size_mb" in alloc
            assert "count" in alloc

    def test_top_allocations_limit(self):
        """Test limiting top allocations."""
        with MemoryProfiler(top_n=10) as profiler:
            data = list(range(1000))
            assert len(data) == 1000

        all_allocs = profiler.top_allocations()
        limited = profiler.top_allocations(2)

        assert len(limited) <= 2
        if len(all_allocs) > 2:
            assert len(limited) < len(all_allocs)

    def test_double_start_warning(self):
        """Test double start logs warning but doesn't crash."""
        profiler = MemoryProfiler()
        profiler.start()
        profiler.start()  # Should warn but not crash
        profiler.stop()


class TestProfilingContextMemory:
    """Tests for profiling_context with memory."""

    def test_memory_profiling_context(self):
        """Test memory profiling via context manager."""
        with profiling_context("memory") as profiler:
            data = list(range(1000))
            assert len(data) == 1000

        assert profiler.result is not None
        assert profiler.result.profile_type == "memory"


class TestCompareMemory:
    """Tests for compare_memory function."""

    def test_compare_memory_increase(self):
        """Test detecting memory increase."""
        baseline = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=100.0,
        )
        current = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=150.0,
        )

        result = compare_memory(baseline, current)

        assert result["baseline_peak_mb"] == 100.0
        assert result["current_peak_mb"] == 150.0
        assert result["diff_mb"] == 50.0
        assert result["diff_pct"] == 50.0
        assert result["regression"] is True  # >20% increase

    def test_compare_memory_decrease(self):
        """Test detecting memory decrease."""
        baseline = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=100.0,
        )
        current = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=80.0,
        )

        result = compare_memory(baseline, current)

        assert result["diff_mb"] == -20.0
        assert result["diff_pct"] == -20.0
        assert result["regression"] is False

    def test_compare_memory_no_regression(self):
        """Test no regression when within threshold."""
        baseline = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=100.0,
        )
        current = ProfileResult(
            profile_type="memory",
            wall_time=1.0,
            peak_memory_mb=115.0,  # 15% increase, under 20% threshold
        )

        result = compare_memory(baseline, current)

        assert result["diff_pct"] == 15.0
        assert result["regression"] is False
