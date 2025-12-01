"""Pytest plugin for performance profiling.

Usage:
    # In conftest.py
    pytest_plugins = ["edm.profiling.pytest_plugin"]

    # In tests
    def test_performance(profile_cpu):
        result = expensive_operation()
        assert profile_cpu.result.wall_time < 5.0

    def test_memory(profile_memory):
        data = load_large_data()
        assert profile_memory.result.peak_memory_mb < 500

    @pytest.mark.benchmark
    def test_with_baseline(benchmark_profile):
        analyze_track(filepath)
        # Automatically compared against baseline
"""

from pathlib import Path
from typing import Any, Generator

import pytest

from edm.profiling.baseline import BaselineStore, compare_baseline, create_baseline
from edm.profiling.cpu import CPUProfiler
from edm.profiling.memory import MemoryProfiler


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add profiling options to pytest."""
    group = parser.getgroup("profiling", "Performance profiling options")

    group.addoption(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling for marked tests",
    )

    group.addoption(
        "--profile-baseline",
        type=str,
        default=None,
        help="Compare against named baseline",
    )

    group.addoption(
        "--profile-save",
        type=str,
        default=None,
        help="Save results as named baseline",
    )

    group.addoption(
        "--profile-dir",
        type=str,
        default="benchmarks/baselines",
        help="Directory for baseline storage",
    )

    group.addoption(
        "--profile-threshold",
        type=float,
        default=20.0,
        help="Regression threshold percentage (default: 20)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a performance benchmark",
    )
    config.addinivalue_line(
        "markers",
        "profile_cpu: enable CPU profiling for this test",
    )
    config.addinivalue_line(
        "markers",
        "profile_memory: enable memory profiling for this test",
    )


@pytest.fixture
def profile_cpu() -> Generator[CPUProfiler, None, None]:
    """Fixture that provides a CPU profiler.

    Example:
        def test_performance(profile_cpu):
            with profile_cpu:
                result = expensive_operation()
            assert profile_cpu.result.wall_time < 5.0
    """
    profiler = CPUProfiler()
    yield profiler


@pytest.fixture
def profile_memory() -> Generator[MemoryProfiler, None, None]:
    """Fixture that provides a memory profiler.

    Example:
        def test_memory(profile_memory):
            with profile_memory:
                data = load_large_data()
            assert profile_memory.result.peak_memory_mb < 500
    """
    profiler = MemoryProfiler()
    yield profiler


class BenchmarkProfiler:
    """Combined profiler for benchmark tests."""

    def __init__(
        self,
        store: BaselineStore,
        baseline_name: str | None = None,
        threshold: float = 0.20,
    ) -> None:
        self._store = store
        self._baseline_name = baseline_name
        self._threshold = threshold
        self._cpu_profiler = CPUProfiler()
        self._memory_profiler = MemoryProfiler()
        self._results: dict[str, Any] = {}

    def __enter__(self) -> "BenchmarkProfiler":
        self._cpu_profiler.start()
        self._memory_profiler.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._memory_profiler.is_running:
            self._results["memory"] = self._memory_profiler.stop()
        if self._cpu_profiler.is_running:
            self._results["cpu"] = self._cpu_profiler.stop()

    @property
    def cpu_result(self) -> Any:
        """Get CPU profiling result."""
        return self._results.get("cpu")

    @property
    def memory_result(self) -> Any:
        """Get memory profiling result."""
        return self._results.get("memory")

    def compare_baseline(self, test_name: str) -> list[Any] | None:
        """Compare results against baseline if configured."""
        if self._baseline_name is None:
            return None

        baseline = self._store.load(self._baseline_name)
        if baseline is None:
            return None

        # Build current profiles dict
        current = {}
        if self.cpu_result:
            current[f"{test_name}_cpu"] = self.cpu_result
        if self.memory_result:
            current[f"{test_name}_memory"] = self.memory_result

        return compare_baseline(baseline, current, self._threshold)


@pytest.fixture
def benchmark_profile(request: pytest.FixtureRequest) -> Generator[BenchmarkProfiler, None, None]:
    """Fixture for benchmark tests with optional baseline comparison.

    Example:
        @pytest.mark.benchmark
        def test_analyze_performance(benchmark_profile):
            with benchmark_profile:
                analyze_track(filepath)
            assert benchmark_profile.cpu_result.wall_time < 10.0
    """
    config = request.config

    baseline_dir = Path(config.getoption("--profile-dir"))
    baseline_name = config.getoption("--profile-baseline")
    threshold = config.getoption("--profile-threshold") / 100

    store = BaselineStore(baseline_dir)
    profiler = BenchmarkProfiler(store, baseline_name, threshold)

    yield profiler

    # After test, optionally save as baseline
    save_name = config.getoption("--profile-save")
    if save_name and profiler._results:
        test_name = request.node.name
        profiles = {}
        if profiler.cpu_result:
            profiles[f"{test_name}_cpu"] = profiler.cpu_result
        if profiler.memory_result:
            profiles[f"{test_name}_memory"] = profiler.memory_result

        if profiles:
            baseline = create_baseline(save_name, profiles)
            store.save(save_name, baseline)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip benchmark tests unless --profile is specified."""
    if config.getoption("--profile"):
        return

    skip_benchmark = pytest.mark.skip(reason="need --profile option to run benchmarks")

    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
