"""Tests for CPU profiler."""

import tempfile
from pathlib import Path

import pytest

from edm.profiling.cpu import CPUProfiler, profile_function


class TestCPUProfiler:
    """Tests for CPUProfiler class."""

    def test_basic_profiling(self):
        """Test basic CPU profiling."""
        profiler = CPUProfiler()
        profiler.start()

        # Do some work
        total = sum(i * i for i in range(10000))
        assert total > 0

        result = profiler.stop()

        assert result is not None
        assert result.profile_type == "cpu"
        assert result.wall_time > 0
        assert result.cpu_time >= 0
        assert len(result.function_stats) > 0

    def test_context_manager(self):
        """Test profiler as context manager."""
        with CPUProfiler() as profiler:
            # Do some work
            data = [x**2 for x in range(1000)]
            assert len(data) == 1000

        assert profiler.result is not None
        assert profiler.result.wall_time > 0
        assert not profiler.is_running

    def test_is_running_state(self):
        """Test is_running property."""
        profiler = CPUProfiler()
        assert not profiler.is_running

        profiler.start()
        assert profiler.is_running

        profiler.stop()
        assert not profiler.is_running

    def test_stop_without_start_raises(self):
        """Test stopping without starting raises error."""
        profiler = CPUProfiler()
        with pytest.raises(RuntimeError, match="not running"):
            profiler.stop()

    def test_double_start_warning(self):
        """Test double start logs warning but doesn't crash."""
        profiler = CPUProfiler()
        profiler.start()
        profiler.start()  # Should warn but not crash
        profiler.stop()

    def test_function_stats_extraction(self):
        """Test that function stats are extracted correctly."""

        def inner_function():
            return sum(range(100))

        def outer_function():
            total = 0
            for _ in range(10):
                total += inner_function()
            return total

        with CPUProfiler() as profiler:
            result = outer_function()
            assert result > 0

        # Should have captured function calls
        assert len(profiler.result.function_stats) > 0

        # Find our functions in stats
        func_names = [fs.name for fs in profiler.result.function_stats]
        # At least some functions should be captured
        assert any("inner_function" in name or "outer_function" in name for name in func_names)

    def test_top_functions(self):
        """Test getting top functions."""

        def slow_function():
            return sum(i * i for i in range(10000))

        def fast_function():
            return 42

        with CPUProfiler() as profiler:
            slow_function()
            for _ in range(100):
                fast_function()

        top = profiler.result.top_functions(5)
        assert len(top) <= 5
        # Top functions should be sorted by cumulative time
        if len(top) >= 2:
            assert top[0].cumulative_time >= top[1].cumulative_time

    def test_get_stats_string(self):
        """Test getting stats as string."""
        with CPUProfiler() as profiler:
            sum(range(100))

        stats_str = profiler.get_stats_string(10)
        assert isinstance(stats_str, str)
        assert len(stats_str) > 0
        assert "function calls" in stats_str.lower() or "cumtime" in stats_str.lower()

    def test_save_stats(self):
        """Test saving stats to file."""
        with CPUProfiler() as profiler:
            sum(range(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "test.prof"
            profiler.save_stats(stats_path)
            assert stats_path.exists()
            assert stats_path.stat().st_size > 0

    def test_save_stats_creates_directory(self):
        """Test save_stats creates parent directories."""
        with CPUProfiler() as profiler:
            sum(range(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "nested" / "dir" / "test.prof"
            profiler.save_stats(stats_path)
            assert stats_path.exists()


class TestProfileFunction:
    """Tests for profile_function helper."""

    def test_basic_usage(self):
        """Test profiling a function."""

        def my_func(n):
            return sum(range(n))

        result, profile = profile_function(my_func, 1000)

        assert result == 499500
        assert profile is not None
        assert profile.wall_time > 0

    def test_with_kwargs(self):
        """Test profiling with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result, profile = profile_function(greet, "World", greeting="Hi")

        assert result == "Hi, World!"
        assert profile.profile_type == "cpu"

    def test_exception_propagates(self):
        """Test that exceptions in profiled code propagate."""

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            profile_function(failing_func)
