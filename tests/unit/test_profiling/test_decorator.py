"""Tests for profiling decorator."""

import os
import tempfile
from pathlib import Path

import pytest

from edm.profiling.decorator import profile


class TestProfileDecorator:
    """Tests for @profile decorator."""

    def test_disabled_by_default(self):
        """Test that profiling is disabled by default."""
        call_count = 0

        @profile()
        def my_func():
            nonlocal call_count
            call_count += 1
            return 42

        # Ensure env var is not set
        os.environ.pop("EDM_PROFILING", None)

        result = my_func()

        assert result == 42
        assert call_count == 1

    def test_enabled_via_parameter(self):
        """Test enabling profiling via parameter."""

        @profile(enabled=True)
        def my_func():
            return sum(range(1000))

        result = my_func()

        assert result == sum(range(1000))

    def test_enabled_via_env_var(self):
        """Test enabling profiling via environment variable."""
        original = os.environ.get("EDM_PROFILING")

        try:
            os.environ["EDM_PROFILING"] = "1"

            @profile()
            def my_func():
                return list(range(100))

            result = my_func()

            assert result == list(range(100))
        finally:
            if original is None:
                os.environ.pop("EDM_PROFILING", None)
            else:
                os.environ["EDM_PROFILING"] = original

    def test_env_var_values(self):
        """Test various env var values for enabling."""
        original = os.environ.get("EDM_PROFILING")

        try:
            for value in ["1", "true", "True", "TRUE", "yes", "YES"]:
                os.environ["EDM_PROFILING"] = value

                @profile()
                def my_func():
                    return 1

                # Should not raise
                result = my_func()
                assert result == 1
        finally:
            if original is None:
                os.environ.pop("EDM_PROFILING", None)
            else:
                os.environ["EDM_PROFILING"] = original

    def test_cpu_profiling(self):
        """Test CPU profiling mode."""

        @profile(profile_type="cpu", enabled=True)
        def compute():
            return sum(i * i for i in range(10000))

        result = compute()

        assert result == sum(i * i for i in range(10000))

    def test_memory_profiling(self):
        """Test memory profiling mode."""

        @profile(profile_type="memory", enabled=True)
        def allocate():
            return list(range(10000))

        result = allocate()

        assert len(result) == 10000

    def test_both_profiling(self):
        """Test combined CPU and memory profiling."""

        @profile(profile_type="both", enabled=True)
        def mixed():
            data = list(range(5000))
            return sum(data)

        result = mixed()

        assert result == sum(range(5000))

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @profile()
        def documented_func():
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_handles_exceptions(self):
        """Test that exceptions are propagated correctly."""

        @profile(enabled=True)
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    def test_output_dir_creates_files(self):
        """Test that output_dir creates profile files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            @profile(enabled=True, output_dir=output_dir)
            def my_func():
                return sum(range(100))

            my_func()

            # Should have created at least one file
            files = list(output_dir.glob("*.json"))
            assert len(files) >= 1

            # Check file contains valid JSON
            import json

            with open(files[0]) as f:
                data = json.load(f)
                assert "profile_type" in data
                assert "wall_time" in data

    def test_output_dir_with_both_profiles(self):
        """Test that both profiles create separate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            @profile(profile_type="both", enabled=True, output_dir=output_dir)
            def my_func():
                return list(range(100))

            my_func()

            files = list(output_dir.glob("*.json"))
            assert len(files) == 2

            # Should have cpu and memory files
            filenames = [f.name for f in files]
            assert any("cpu" in name for name in filenames)
            assert any("memory" in name for name in filenames)

    def test_with_args_and_kwargs(self):
        """Test that decorated function handles args/kwargs correctly."""

        @profile(enabled=True)
        def add(a, b, c=0):
            return a + b + c

        assert add(1, 2) == 3
        assert add(1, 2, c=3) == 6
        assert add(1, 2, 3) == 6

    def test_top_n_parameter(self):
        """Test that top_n parameter is passed to profiler."""

        @profile(enabled=True, top_n=5)
        def my_func():
            return [list(range(100)) for _ in range(10)]

        result = my_func()
        assert len(result) == 10
