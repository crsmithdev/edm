"""Tests for parallel processing utilities."""

import time
from unittest.mock import MagicMock, patch

import pytest
from edm.processing.parallel import (
    MAX_WORKERS,
    ParallelProcessor,
    get_default_workers,
    validate_worker_count,
)


# Simple worker function for testing (must be top-level for pickling)
def _square_worker(x: int) -> int:
    """Square a number."""
    return x * x


def _slow_worker(x: int) -> int:
    """Slow worker that sleeps briefly."""
    time.sleep(0.01)
    return x * x


def _error_worker(x: int) -> int:
    """Worker that raises an error for negative inputs."""
    if x < 0:
        raise ValueError(f"Negative input: {x}")
    return x * x


class TestGetDefaultWorkers:
    """Tests for get_default_workers function."""

    def test_returns_at_least_one(self):
        """Test that it returns at least 1 worker."""
        result = get_default_workers()
        assert result >= 1

    @patch("os.cpu_count", return_value=4)
    def test_returns_cpu_count_minus_one(self, mock_cpu_count):
        """Test that it returns cpu_count - 1."""
        result = get_default_workers()
        assert result == 3

    @patch("os.cpu_count", return_value=1)
    def test_minimum_one_with_single_cpu(self, mock_cpu_count):
        """Test minimum of 1 worker with single CPU."""
        result = get_default_workers()
        assert result == 1

    @patch("os.cpu_count", return_value=None)
    def test_handles_none_cpu_count(self, mock_cpu_count):
        """Test handling when cpu_count returns None."""
        result = get_default_workers()
        assert result == 1


class TestValidateWorkerCount:
    """Tests for validate_worker_count function."""

    def test_valid_count_unchanged(self):
        """Test that valid worker count is returned unchanged."""
        result = validate_worker_count(4)
        assert result == 4

    def test_raises_for_zero(self):
        """Test that 0 workers raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            validate_worker_count(0)

    def test_raises_for_negative(self):
        """Test that negative workers raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            validate_worker_count(-1)

    def test_caps_at_max_workers(self):
        """Test that worker count is capped at MAX_WORKERS."""
        result = validate_worker_count(MAX_WORKERS + 10)
        assert result == MAX_WORKERS

    @patch("os.cpu_count", return_value=4)
    def test_warns_when_exceeds_cpu_count(self, mock_cpu_count, caplog):
        """Test warning when workers exceed cpu count."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = validate_worker_count(8)
        assert result == 8  # Still returns the value (under MAX_WORKERS)


class TestParallelProcessor:
    """Tests for ParallelProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = ParallelProcessor(
            worker_fn=_square_worker,
            workers=4,
        )
        assert processor.workers == 4
        assert processor.worker_fn == _square_worker

    def test_init_validates_workers(self):
        """Test that init validates worker count."""
        with pytest.raises(ValueError):
            ParallelProcessor(worker_fn=_square_worker, workers=0)

    def test_process_empty_list(self):
        """Test processing empty list returns empty list."""
        processor = ParallelProcessor(worker_fn=_square_worker, workers=2)
        result = processor.process([])
        assert result == []

    def test_process_single_worker(self):
        """Test sequential processing with single worker."""
        processor = ParallelProcessor(worker_fn=_square_worker, workers=1)
        result = processor.process([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_process_multiple_workers(self):
        """Test parallel processing with multiple workers."""
        processor = ParallelProcessor(worker_fn=_square_worker, workers=2)
        result = processor.process([1, 2, 3, 4, 5])
        assert result == [1, 4, 9, 16, 25]

    def test_maintains_order(self):
        """Test that results maintain input order."""
        processor = ParallelProcessor(worker_fn=_slow_worker, workers=4)
        result = processor.process([5, 4, 3, 2, 1])
        assert result == [25, 16, 9, 4, 1]

    def test_progress_callback_called(self):
        """Test that progress callback is called."""
        callback = MagicMock()
        processor = ParallelProcessor(
            worker_fn=_square_worker,
            workers=1,
            progress_callback=callback,
        )
        processor.process([1, 2, 3])

        # Should be called once per item
        assert callback.call_count == 3
        callback.assert_any_call(1)
        callback.assert_any_call(2)
        callback.assert_any_call(3)

    def test_progress_callback_parallel(self):
        """Test progress callback with parallel workers."""
        callback = MagicMock()
        processor = ParallelProcessor(
            worker_fn=_square_worker,
            workers=2,
            progress_callback=callback,
        )
        processor.process([1, 2, 3, 4])

        # Should be called once per item
        assert callback.call_count == 4

    def test_terminate(self):
        """Test terminate method."""
        processor = ParallelProcessor(worker_fn=_square_worker, workers=2)
        # Just verify it doesn't error when no pool is running
        processor.terminate()
        assert processor._pool is None


class TestParallelProcessorIntegration:
    """Integration tests for ParallelProcessor."""

    def test_large_batch(self):
        """Test processing a larger batch."""
        processor = ParallelProcessor(worker_fn=_square_worker, workers=4)
        items = list(range(100))
        result = processor.process(items)
        expected = [x * x for x in items]
        assert result == expected

    def test_sequential_vs_parallel_same_results(self):
        """Test that sequential and parallel produce same results."""
        items = list(range(20))

        sequential = ParallelProcessor(worker_fn=_square_worker, workers=1)
        parallel = ParallelProcessor(worker_fn=_square_worker, workers=4)

        seq_result = sequential.process(items)
        par_result = parallel.process(items)

        assert seq_result == par_result
