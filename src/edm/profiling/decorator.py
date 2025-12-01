"""Profiling decorator for easy function instrumentation."""

import functools
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

import structlog

from edm.profiling.base import ProfileResult
from edm.profiling.cpu import CPUProfiler
from edm.profiling.memory import MemoryProfiler

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def profile(
    profile_type: str = "cpu",
    enabled: bool | None = None,
    output_dir: Path | None = None,
    top_n: int = 10,
) -> Callable[[F], F]:
    """Decorator to profile a function.

    Args:
        profile_type: Type of profiling ('cpu', 'memory', 'both').
        enabled: Whether profiling is enabled. If None, checks EDM_PROFILING env var.
        output_dir: Directory to save profile results. If None, results are logged only.
        top_n: Number of top functions/allocations to track.

    Returns:
        Decorated function that profiles when called.

    Example:
        @profile()
        def analyze_track(filepath):
            # ... analysis code ...

        @profile(profile_type="memory", output_dir=Path("profiles"))
        def load_audio(filepath):
            # ... loading code ...

        # Enable via environment variable
        EDM_PROFILING=1 python script.py
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if profiling is enabled
            is_enabled = enabled
            if is_enabled is None:
                is_enabled = os.environ.get("EDM_PROFILING", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )

            if not is_enabled:
                return func(*args, **kwargs)

            func_name = f"{func.__module__}.{func.__qualname__}"
            logger.debug("profiling function", name=func_name, type=profile_type)

            results: dict[str, ProfileResult] = {}

            # Run with appropriate profiler(s)
            if profile_type in ("cpu", "both"):
                cpu_profiler = CPUProfiler()
                cpu_profiler.start()

            if profile_type in ("memory", "both"):
                mem_profiler = MemoryProfiler(top_n=top_n)
                mem_profiler.start()

            try:
                result = func(*args, **kwargs)
            finally:
                if profile_type in ("memory", "both"):
                    mem_result = mem_profiler.stop()
                    results["memory"] = mem_result
                    logger.info(
                        "memory profile",
                        function=func_name,
                        wall_time=f"{mem_result.wall_time:.3f}s",
                        peak_mb=f"{mem_result.peak_memory_mb:.1f}",
                    )

                if profile_type in ("cpu", "both"):
                    cpu_result = cpu_profiler.stop()
                    results["cpu"] = cpu_result
                    logger.info(
                        "cpu profile",
                        function=func_name,
                        wall_time=f"{cpu_result.wall_time:.3f}s",
                        cpu_time=f"{cpu_result.cpu_time:.3f}s",
                    )

                # Save results if output directory specified
                if output_dir is not None:
                    _save_profile_results(func_name, results, output_dir)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _save_profile_results(
    func_name: str,
    results: dict[str, ProfileResult],
    output_dir: Path,
) -> None:
    """Save profile results to files."""
    import json
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from function name and timestamp
    safe_name = func_name.replace(".", "_").replace("<", "").replace(">", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for profile_type, result in results.items():
        filename = f"{safe_name}_{profile_type}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.debug("saved profile", path=str(filepath))
