"""Base classes and data structures for profiling."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator


@dataclass
class FunctionStats:
    """Statistics for a single function.

    Attributes:
        name: Function name (module:function format).
        calls: Number of times the function was called.
        total_time: Total time spent in function (excluding subcalls).
        cumulative_time: Cumulative time (including subcalls).
        callers: List of caller function names.
    """

    name: str
    calls: int
    total_time: float
    cumulative_time: float
    callers: list[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Result of a profiling session.

    Attributes:
        profile_type: Type of profiling ('cpu', 'memory', 'io').
        wall_time: Wall clock time in seconds.
        cpu_time: CPU time in seconds (for CPU profiling).
        peak_memory_mb: Peak memory usage in MB (for memory profiling).
        function_stats: Per-function statistics.
        timestamp: When the profiling was performed.
        metadata: Additional metadata (git commit, etc.).
    """

    profile_type: str
    wall_time: float
    cpu_time: float = 0.0
    peak_memory_mb: float = 0.0
    function_stats: list[FunctionStats] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "profile_type": self.profile_type,
            "wall_time": self.wall_time,
            "cpu_time": self.cpu_time,
            "peak_memory_mb": self.peak_memory_mb,
            "function_stats": [
                {
                    "name": fs.name,
                    "calls": fs.calls,
                    "total_time": fs.total_time,
                    "cumulative_time": fs.cumulative_time,
                    "callers": fs.callers,
                }
                for fs in self.function_stats
            ],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileResult":
        """Create from dictionary."""
        return cls(
            profile_type=data["profile_type"],
            wall_time=data["wall_time"],
            cpu_time=data.get("cpu_time", 0.0),
            peak_memory_mb=data.get("peak_memory_mb", 0.0),
            function_stats=[
                FunctionStats(
                    name=fs["name"],
                    calls=fs["calls"],
                    total_time=fs["total_time"],
                    cumulative_time=fs["cumulative_time"],
                    callers=fs.get("callers", []),
                )
                for fs in data.get("function_stats", [])
            ],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

    def top_functions(self, n: int = 10, by: str = "cumulative_time") -> list[FunctionStats]:
        """Get top N functions by specified metric.

        Args:
            n: Number of functions to return.
            by: Metric to sort by ('cumulative_time', 'total_time', 'calls').

        Returns:
            List of top N functions sorted by metric.
        """
        return sorted(self.function_stats, key=lambda f: getattr(f, by), reverse=True)[:n]


class ProfilerBase(ABC):
    """Abstract base class for profilers.

    Profilers support context manager usage:
        with CPUProfiler() as profiler:
            # code to profile
        result = profiler.result
    """

    def __init__(self) -> None:
        self._result: ProfileResult | None = None
        self._is_running: bool = False

    @property
    def result(self) -> ProfileResult | None:
        """Get the profiling result after stop() is called."""
        return self._result

    @property
    def is_running(self) -> bool:
        """Check if profiler is currently running."""
        return self._is_running

    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        ...

    @abstractmethod
    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        ...

    def __enter__(self) -> "ProfilerBase":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._is_running:
            self.stop()


@contextmanager
def profiling_context(
    profile_type: str = "cpu",
) -> Generator[ProfilerBase, None, None]:
    """Context manager for profiling code blocks.

    Args:
        profile_type: Type of profiling ('cpu', 'memory').

    Yields:
        Profiler instance with results available after context exits.

    Example:
        with profiling_context('cpu') as profiler:
            analyze_bpm(filepath)
        print(f"Took {profiler.result.wall_time:.2f}s")
    """
    from edm.profiling.cpu import CPUProfiler
    from edm.profiling.memory import MemoryProfiler

    profilers: dict[str, type[ProfilerBase]] = {
        "cpu": CPUProfiler,
        "memory": MemoryProfiler,
    }

    profiler_cls = profilers.get(profile_type)
    if profiler_cls is None:
        raise ValueError(f"Unknown profile type: {profile_type}")

    profiler = profiler_cls()
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.stop()
