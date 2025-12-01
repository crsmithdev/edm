"""Tests for pytest plugin."""

import tempfile
from pathlib import Path

from edm.profiling.baseline import BaselineStore
from edm.profiling.pytest_plugin import BenchmarkProfiler


class TestProfileFixtures:
    """Tests for profiling fixtures."""

    def test_profile_cpu_fixture(self, profile_cpu):
        """Test CPU profiler fixture."""
        assert not profile_cpu.is_running

        with profile_cpu:
            _ = sum(range(1000))

        assert profile_cpu.result is not None
        assert profile_cpu.result.profile_type == "cpu"
        assert profile_cpu.result.wall_time > 0

    def test_profile_memory_fixture(self, profile_memory):
        """Test memory profiler fixture."""
        assert not profile_memory.is_running

        with profile_memory:
            data = list(range(10000))

        assert profile_memory.result is not None
        assert profile_memory.result.profile_type == "memory"
        assert len(data) == 10000


class TestBenchmarkProfiler:
    """Tests for BenchmarkProfiler."""

    def test_context_manager(self):
        """Test as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            profiler = BenchmarkProfiler(store)

            with profiler:
                result = sum(range(1000))

            assert profiler.cpu_result is not None
            assert profiler.memory_result is not None
            assert result == sum(range(1000))

    def test_cpu_result(self):
        """Test CPU result access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            profiler = BenchmarkProfiler(store)

            with profiler:
                _ = [i * i for i in range(1000)]

            assert profiler.cpu_result.wall_time > 0
            assert profiler.cpu_result.profile_type == "cpu"

    def test_memory_result(self):
        """Test memory result access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            profiler = BenchmarkProfiler(store)

            with profiler:
                data = list(range(10000))

            assert profiler.memory_result.profile_type == "memory"
            assert len(data) == 10000

    def test_compare_without_baseline(self):
        """Test comparison without baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            profiler = BenchmarkProfiler(store)

            with profiler:
                pass

            result = profiler.compare_baseline("test")
            assert result is None

    def test_compare_with_missing_baseline(self):
        """Test comparison with missing baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            profiler = BenchmarkProfiler(store, baseline_name="nonexistent")

            with profiler:
                pass

            result = profiler.compare_baseline("test")
            assert result is None
