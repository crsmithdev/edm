"""CPU profiling using cProfile."""

import cProfile
import pstats
import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

import structlog

from edm.profiling.base import FunctionStats, ProfilerBase, ProfileResult

logger = structlog.get_logger(__name__)


class CPUProfiler(ProfilerBase):
    """CPU profiler using Python's cProfile module.

    Provides detailed function-level timing information with low overhead.

    Example:
        with CPUProfiler() as profiler:
            analyze_bpm(filepath)

        # Access results
        print(f"Wall time: {profiler.result.wall_time:.2f}s")
        for func in profiler.result.top_functions(5):
            print(f"  {func.name}: {func.cumulative_time:.3f}s")
    """

    def __init__(self) -> None:
        super().__init__()
        self._profiler: cProfile.Profile | None = None
        self._start_time: float = 0.0
        self._stats: pstats.Stats | None = None

    def start(self) -> None:
        """Start CPU profiling."""
        if self._is_running:
            logger.warning("profiler already running")
            return

        self._profiler = cProfile.Profile()
        self._start_time = time.perf_counter()
        self._profiler.enable()
        self._is_running = True
        logger.debug("cpu profiler started")

    def stop(self) -> ProfileResult:
        """Stop profiling and return results.

        Returns:
            ProfileResult with CPU timing data and function statistics.
        """
        if not self._is_running or self._profiler is None:
            raise RuntimeError("Profiler not running")

        self._profiler.disable()
        wall_time = time.perf_counter() - self._start_time
        self._is_running = False

        # Create stats object
        self._stats = pstats.Stats(self._profiler)
        self._stats.sort_stats("cumulative")

        # Extract function statistics
        function_stats = self._extract_function_stats()

        # Calculate total CPU time
        cpu_time = sum(fs.total_time for fs in function_stats)

        self._result = ProfileResult(
            profile_type="cpu",
            wall_time=wall_time,
            cpu_time=cpu_time,
            function_stats=function_stats,
        )

        logger.debug(
            "cpu profiler stopped",
            wall_time=f"{wall_time:.3f}s",
            cpu_time=f"{cpu_time:.3f}s",
            functions=len(function_stats),
        )

        return self._result

    def _extract_function_stats(self) -> list[FunctionStats]:
        """Extract function statistics from cProfile stats."""
        if self._stats is None:
            return []

        stats_list = []
        # pstats.Stats.stats is a dict: (filename, lineno, funcname) -> (ncalls, tottime, cumtime, ...)
        # Type stubs don't include this internal attribute
        for key, value in self._stats.stats.items():  # type: ignore[attr-defined]
            filename, lineno, funcname = key
            ncalls, totcalls, tottime, cumtime, callers = value

            # Format function name
            if filename.startswith("<"):
                name = f"{filename}:{funcname}"
            else:
                # Shorten path for readability
                short_file = Path(filename).name
                name = f"{short_file}:{lineno}:{funcname}"

            # Extract caller names
            caller_names = []
            for caller_key in callers:
                c_file, c_line, c_func = caller_key
                c_short = Path(c_file).name if not c_file.startswith("<") else c_file
                caller_names.append(f"{c_short}:{c_func}")

            stats_list.append(
                FunctionStats(
                    name=name,
                    calls=ncalls,
                    total_time=tottime,
                    cumulative_time=cumtime,
                    callers=caller_names[:5],  # Limit callers for readability
                )
            )

        return stats_list

    def print_stats(self, n: int = 20) -> None:
        """Print top N functions by cumulative time.

        Args:
            n: Number of functions to print.
        """
        if self._stats is None:
            logger.warning("no stats available, run profiler first")
            return

        self._stats.print_stats(n)

    def get_stats_string(self, n: int = 20) -> str:
        """Get stats as a string.

        Args:
            n: Number of functions to include.

        Returns:
            Formatted statistics string.
        """
        if self._stats is None:
            return "No stats available"

        stream = StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(n)
        return stream.getvalue()

    def save_stats(self, filepath: Path) -> None:
        """Save raw cProfile stats to file.

        Args:
            filepath: Path to save stats file (.prof extension recommended).
        """
        if self._profiler is None:
            raise RuntimeError("No profiler data available")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._profiler.dump_stats(str(filepath))
        logger.info("saved cpu profile stats", path=str(filepath))


def profile_function(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, ProfileResult]:
    """Profile a single function call.

    Args:
        func: Function to profile.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Tuple of (function return value, ProfileResult).

    Example:
        result, profile = profile_function(analyze_bpm, filepath)
        print(f"BPM: {result.bpm}, Time: {profile.wall_time:.2f}s")
    """
    with CPUProfiler() as profiler:
        result = func(*args, **kwargs)
    assert profiler.result is not None  # Always set after context manager
    return result, profiler.result
