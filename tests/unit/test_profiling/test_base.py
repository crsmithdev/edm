"""Tests for profiling base classes."""

from datetime import datetime

import pytest

from edm.profiling.base import FunctionStats, ProfileResult, profiling_context


class TestFunctionStats:
    """Tests for FunctionStats dataclass."""

    def test_creation(self):
        """Test basic creation."""
        stats = FunctionStats(
            name="module.py:42:my_function",
            calls=100,
            total_time=1.5,
            cumulative_time=2.3,
            callers=["caller1", "caller2"],
        )
        assert stats.name == "module.py:42:my_function"
        assert stats.calls == 100
        assert stats.total_time == 1.5
        assert stats.cumulative_time == 2.3
        assert len(stats.callers) == 2

    def test_default_callers(self):
        """Test default empty callers list."""
        stats = FunctionStats(
            name="test",
            calls=1,
            total_time=0.1,
            cumulative_time=0.1,
        )
        assert stats.callers == []


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = ProfileResult(
            profile_type="cpu",
            wall_time=1.5,
            cpu_time=1.2,
        )
        assert result.profile_type == "cpu"
        assert result.wall_time == 1.5
        assert result.cpu_time == 1.2
        assert result.peak_memory_mb == 0.0
        assert result.function_stats == []
        assert isinstance(result.timestamp, datetime)

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = FunctionStats(
            name="test_func",
            calls=10,
            total_time=0.5,
            cumulative_time=0.8,
        )
        result = ProfileResult(
            profile_type="cpu",
            wall_time=1.0,
            cpu_time=0.8,
            function_stats=[stats],
            metadata={"commit": "abc123"},
        )

        data = result.to_dict()

        assert data["profile_type"] == "cpu"
        assert data["wall_time"] == 1.0
        assert data["cpu_time"] == 0.8
        assert len(data["function_stats"]) == 1
        assert data["function_stats"][0]["name"] == "test_func"
        assert data["metadata"]["commit"] == "abc123"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "profile_type": "cpu",
            "wall_time": 2.5,
            "cpu_time": 2.0,
            "peak_memory_mb": 100.5,
            "function_stats": [
                {
                    "name": "func1",
                    "calls": 50,
                    "total_time": 1.0,
                    "cumulative_time": 1.5,
                    "callers": ["main"],
                }
            ],
            "timestamp": "2025-01-01T12:00:00",
            "metadata": {"version": "1.0"},
        }

        result = ProfileResult.from_dict(data)

        assert result.profile_type == "cpu"
        assert result.wall_time == 2.5
        assert result.cpu_time == 2.0
        assert result.peak_memory_mb == 100.5
        assert len(result.function_stats) == 1
        assert result.function_stats[0].name == "func1"
        assert result.metadata["version"] == "1.0"

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = ProfileResult(
            profile_type="memory",
            wall_time=3.0,
            peak_memory_mb=512.0,
            function_stats=[
                FunctionStats("f1", 10, 0.1, 0.2),
                FunctionStats("f2", 20, 0.2, 0.4),
            ],
        )

        restored = ProfileResult.from_dict(original.to_dict())

        assert restored.profile_type == original.profile_type
        assert restored.wall_time == original.wall_time
        assert restored.peak_memory_mb == original.peak_memory_mb
        assert len(restored.function_stats) == 2

    def test_top_functions(self):
        """Test top_functions sorting."""
        result = ProfileResult(
            profile_type="cpu",
            wall_time=1.0,
            function_stats=[
                FunctionStats("slow", 1, 0.1, 0.5),
                FunctionStats("fast", 100, 0.01, 0.1),
                FunctionStats("medium", 10, 0.05, 0.3),
            ],
        )

        # By cumulative time (default)
        top = result.top_functions(2)
        assert len(top) == 2
        assert top[0].name == "slow"
        assert top[1].name == "medium"

        # By calls
        top_calls = result.top_functions(2, by="calls")
        assert top_calls[0].name == "fast"

        # By total time
        top_total = result.top_functions(2, by="total_time")
        assert top_total[0].name == "slow"


class TestProfilingContext:
    """Tests for profiling_context context manager."""

    def test_cpu_profiling(self):
        """Test CPU profiling context."""
        with profiling_context("cpu") as profiler:
            # Do some work
            total = sum(range(1000))
            assert total == 499500

        assert profiler.result is not None
        assert profiler.result.profile_type == "cpu"
        assert profiler.result.wall_time > 0

    def test_invalid_profile_type(self):
        """Test invalid profile type raises error."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            with profiling_context("invalid"):
                pass

    def test_profiler_not_running_after_context(self):
        """Test profiler stops after context exits."""
        with profiling_context("cpu") as profiler:
            assert profiler.is_running

        assert not profiler.is_running
