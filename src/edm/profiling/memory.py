"""Memory profiling using tracemalloc."""

import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from edm.profiling.base import FunctionStats, ProfilerBase, ProfileResult

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """A memory allocation snapshot.

    Attributes:
        timestamp: When the snapshot was taken.
        current_mb: Current memory usage in MB.
        peak_mb: Peak memory usage since start in MB.
        top_allocations: Top memory allocations by size.
    """

    timestamp: float
    current_mb: float
    peak_mb: float
    top_allocations: list[dict[str, Any]] = field(default_factory=list)


class MemoryProfiler(ProfilerBase):
    """Memory profiler using Python's tracemalloc module.

    Tracks memory allocations and identifies memory-heavy code paths.

    Example:
        with MemoryProfiler() as profiler:
            data = load_large_file(filepath)
            process(data)

        print(f"Peak memory: {profiler.result.peak_memory_mb:.1f} MB")
        for alloc in profiler.top_allocations(5):
            print(f"  {alloc['file']}:{alloc['line']}: {alloc['size_mb']:.1f} MB")
    """

    def __init__(self, top_n: int = 10) -> None:
        """Initialize memory profiler.

        Args:
            top_n: Number of top allocations to track.
        """
        super().__init__()
        self._top_n = top_n
        self._start_time: float = 0.0
        self._snapshots: list[MemorySnapshot] = []
        self._was_tracing: bool = False

    def start(self) -> None:
        """Start memory profiling."""
        if self._is_running:
            logger.warning("memory profiler already running")
            return

        self._start_time = time.perf_counter()
        self._snapshots = []

        # Check if tracemalloc is already running
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start()

        self._is_running = True
        logger.debug("memory profiler started")

    def stop(self) -> ProfileResult:
        """Stop profiling and return results.

        Returns:
            ProfileResult with memory usage data.
        """
        if not self._is_running:
            raise RuntimeError("Memory profiler not running")

        wall_time = time.perf_counter() - self._start_time

        # Take final snapshot
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()

        # Stop tracemalloc if we started it
        if not self._was_tracing:
            tracemalloc.stop()

        self._is_running = False

        # Extract top allocations
        top_stats = snapshot.statistics("lineno")[: self._top_n]
        function_stats = []
        top_allocations = []

        for stat in top_stats:
            frame = stat.traceback[0] if stat.traceback else None
            if frame:
                filename = Path(frame.filename).name
                name = f"{filename}:{frame.lineno}"
                size_mb = stat.size / (1024 * 1024)

                function_stats.append(
                    FunctionStats(
                        name=name,
                        calls=stat.count,
                        total_time=0.0,  # Memory profiler doesn't track time
                        cumulative_time=size_mb,  # Use cumulative_time for size
                    )
                )

                top_allocations.append(
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "size_mb": size_mb,
                        "count": stat.count,
                    }
                )

        self._result = ProfileResult(
            profile_type="memory",
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 * 1024),
            function_stats=function_stats,
            metadata={"top_allocations": top_allocations},
        )

        logger.debug(
            "memory profiler stopped",
            wall_time=f"{wall_time:.3f}s",
            peak_mb=f"{self._result.peak_memory_mb:.1f}",
            current_mb=f"{current / (1024 * 1024):.1f}",
        )

        return self._result

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot during profiling.

        Returns:
            MemorySnapshot with current memory state.

        Raises:
            RuntimeError: If profiler is not running.
        """
        if not self._is_running:
            raise RuntimeError("Memory profiler not running")

        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[: self._top_n]

        top_allocations = []
        for stat in top_stats:
            frame = stat.traceback[0] if stat.traceback else None
            if frame:
                top_allocations.append(
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                    }
                )

        mem_snapshot = MemorySnapshot(
            timestamp=time.perf_counter() - self._start_time,
            current_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
            top_allocations=top_allocations,
        )

        self._snapshots.append(mem_snapshot)
        return mem_snapshot

    def top_allocations(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get top memory allocations from result.

        Args:
            n: Number of allocations to return (default: all tracked).

        Returns:
            List of allocation dicts with file, line, size_mb, count.
        """
        if self._result is None:
            return []

        allocations: list[dict[str, Any]] = self._result.metadata.get("top_allocations", [])
        if n is not None:
            return allocations[:n]
        return allocations

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        """Get all snapshots taken during profiling."""
        return self._snapshots.copy()


def compare_memory(baseline: ProfileResult, current: ProfileResult) -> dict[str, Any]:
    """Compare memory usage between two profile results.

    Args:
        baseline: Baseline profile result.
        current: Current profile result to compare.

    Returns:
        Comparison dict with peak_diff_mb, peak_diff_pct, and regression flag.
    """
    baseline_peak = baseline.peak_memory_mb
    current_peak = current.peak_memory_mb

    diff_mb = current_peak - baseline_peak
    diff_pct = (diff_mb / baseline_peak * 100) if baseline_peak > 0 else 0.0

    return {
        "baseline_peak_mb": baseline_peak,
        "current_peak_mb": current_peak,
        "diff_mb": diff_mb,
        "diff_pct": diff_pct,
        "regression": diff_pct > 20.0,  # >20% increase is regression
    }
