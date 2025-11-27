"""Parallel processing utilities using multiprocessing."""

import logging
import multiprocessing
import os
import signal
from multiprocessing.pool import Pool
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

# Environment variable for worker process log level
_LOG_LEVEL_ENV_VAR = "EDM_WORKER_LOG_LEVEL"

# Maximum allowed workers to prevent resource exhaustion
MAX_WORKERS = 32


def set_worker_log_level(level: str) -> None:
    """Set the log level for worker processes.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    os.environ[_LOG_LEVEL_ENV_VAR] = level.upper()


def get_worker_log_level() -> str:
    """Get the configured log level for worker processes.

    Returns:
        Log level string (defaults to WARNING).
    """
    return os.environ.get(_LOG_LEVEL_ENV_VAR, "WARNING")


def get_default_workers() -> int:
    """Get default worker count (CPU count - 1, minimum 1).

    Leaves one core for system operations and UI responsiveness.

    Returns:
        Default number of workers.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def validate_worker_count(workers: int) -> int:
    """Validate and adjust worker count.

    Args:
        workers: Requested number of workers.

    Returns:
        Validated worker count.

    Raises:
        ValueError: If workers < 1.
    """
    if workers < 1:
        raise ValueError("Worker count must be at least 1")

    cpu_count = os.cpu_count() or 1

    if workers > cpu_count:
        logger.warning(
            "worker count exceeds cpu count",
            workers=workers,
            cpu_count=cpu_count,
        )

    if workers > MAX_WORKERS:
        logger.warning(
            "capping worker count at maximum",
            requested=workers,
            max_workers=MAX_WORKERS,
        )
        return MAX_WORKERS

    return workers


def _init_worker():
    """Initialize worker process.

    Sets up signal handlers to ignore SIGINT in workers,
    allowing the main process to handle Ctrl+C gracefully.
    Also configures logging to match the main process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure logging for worker process
    from edm.logging import configure_logging

    configure_logging(level=get_worker_log_level())


class ParallelProcessor:
    """Process items in parallel using multiprocessing.

    Provides a simple interface for parallel execution of CPU-bound tasks
    with progress tracking and graceful shutdown handling.

    Attributes:
        worker_fn: Function to execute for each item.
        workers: Number of worker processes.
        show_progress: Whether to track progress via callback.
    """

    def __init__(
        self,
        worker_fn: Callable[[Any], Any],
        workers: int = 1,
        progress_callback: Callable[[int], None] | None = None,
    ):
        """Initialize parallel processor.

        Args:
            worker_fn: Function to execute for each item. Must be picklable
                (top-level function, not a lambda or method).
            workers: Number of worker processes (default: 1).
            progress_callback: Optional callback called after each item completes.
                Receives the number of completed items.
        """
        self.worker_fn = worker_fn
        self.workers = validate_worker_count(workers)
        self.progress_callback = progress_callback
        self._pool: Pool | None = None

    def process(self, items: list[Any]) -> list[Any]:
        """Process items in parallel.

        Args:
            items: List of items to process. Each item is passed to worker_fn.

        Returns:
            List of results in the same order as input items.

        Raises:
            KeyboardInterrupt: If interrupted, terminates workers and re-raises.
        """
        if not items:
            return []

        # Sequential path for single worker
        if self.workers == 1:
            return self._process_sequential(items)

        return self._process_parallel(items)

    def _process_sequential(self, items: list[Any]) -> list[Any]:
        """Process items sequentially.

        Args:
            items: List of items to process.

        Returns:
            List of results.
        """
        results = []
        for i, item in enumerate(items):
            result = self.worker_fn(item)
            results.append(result)
            if self.progress_callback:
                self.progress_callback(i + 1)
        return results

    def _process_parallel(self, items: list[Any]) -> list[Any]:
        """Process items in parallel using multiprocessing.Pool.

        Args:
            items: List of items to process.

        Returns:
            List of results in original order.
        """
        logger.info(
            "starting parallel processing",
            workers=self.workers,
            items=len(items),
        )

        results = []
        try:
            # Use 'spawn' context for cross-platform compatibility
            # (fork can cause issues with certain libraries)
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.workers,
                initializer=_init_worker,
            )

            # Use imap to get results as they complete while maintaining order
            completed = 0
            for result in self._pool.imap(self.worker_fn, items):
                results.append(result)
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed)

            self._pool.close()
            self._pool.join()

        except KeyboardInterrupt:
            logger.warning("keyboard interrupt received, terminating workers")
            if self._pool:
                self._pool.terminate()
                self._pool.join()
            raise

        except Exception as e:
            logger.error("parallel processing failed", error=str(e))
            if self._pool:
                self._pool.terminate()
                self._pool.join()
            raise

        finally:
            self._pool = None

        logger.info("parallel processing complete", results=len(results))
        return results

    def terminate(self):
        """Terminate all workers immediately."""
        if self._pool:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
