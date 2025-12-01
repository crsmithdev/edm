"""Tests for profile reports."""

import json
import tempfile
from pathlib import Path

from edm.profiling.base import ProfileResult
from edm.profiling.baseline import ComparisonResult
from edm.profiling.reports import (
    ProfileReport,
    format_comparison_json,
    format_comparison_text,
    format_profile_json,
    format_profile_text,
    save_report,
)


class TestFormatProfileText:
    """Tests for format_profile_text."""

    def test_basic_format(self):
        """Test basic text formatting."""
        result = ProfileResult(
            profile_type="cpu",
            wall_time=1.234,
            cpu_time=1.1,
        )

        text = format_profile_text(result)

        assert "cpu" in text
        assert "1.234" in text
        assert "1.1" in text.lower() or "1.100" in text

    def test_with_memory(self):
        """Test formatting with memory info."""
        result = ProfileResult(
            profile_type="memory",
            wall_time=2.0,
            peak_memory_mb=512.5,
        )

        text = format_profile_text(result)

        assert "512.5" in text
        assert "MB" in text

    def test_custom_title(self):
        """Test custom title."""
        result = ProfileResult(profile_type="cpu", wall_time=1.0)

        text = format_profile_text(result, title="My Custom Title")

        assert "My Custom Title" in text


class TestFormatComparisonText:
    """Tests for format_comparison_text."""

    def test_with_regression(self):
        """Test formatting with regression."""
        results = [
            ComparisonResult(
                profile_name="test",
                metric="wall_time",
                baseline_value=1.0,
                current_value=1.5,
                diff_absolute=0.5,
                diff_percent=50.0,
                is_regression=True,
                threshold=20.0,
            )
        ]

        text = format_comparison_text(results)

        assert "REGRESSION" in text.upper()
        assert "50" in text
        assert "test" in text

    def test_with_improvement(self):
        """Test formatting with improvement."""
        results = [
            ComparisonResult(
                profile_name="test",
                metric="wall_time",
                baseline_value=1.0,
                current_value=0.8,
                diff_absolute=-0.2,
                diff_percent=-20.0,
                is_regression=False,
                threshold=20.0,
            )
        ]

        text = format_comparison_text(results)

        assert "Improvement" in text
        assert "-20" in text

    def test_summary_counts(self):
        """Test summary counts."""
        results = [
            ComparisonResult(
                profile_name="fast",
                metric="wall_time",
                baseline_value=1.0,
                current_value=0.5,
                diff_absolute=-0.5,
                diff_percent=-50.0,
                is_regression=False,
                threshold=20.0,
            ),
            ComparisonResult(
                profile_name="slow",
                metric="wall_time",
                baseline_value=1.0,
                current_value=1.5,
                diff_absolute=0.5,
                diff_percent=50.0,
                is_regression=True,
                threshold=20.0,
            ),
        ]

        text = format_comparison_text(results)

        assert "Regressions: 1" in text
        assert "Improvements: 1" in text


class TestFormatJson:
    """Tests for JSON formatting."""

    def test_profile_json(self):
        """Test profile JSON formatting."""
        result = ProfileResult(
            profile_type="cpu",
            wall_time=1.5,
            cpu_time=1.2,
        )

        json_str = format_profile_json(result)
        data = json.loads(json_str)

        assert data["profile_type"] == "cpu"
        assert data["wall_time"] == 1.5

    def test_comparison_json(self):
        """Test comparison JSON formatting."""
        results = [
            ComparisonResult(
                profile_name="test",
                metric="wall_time",
                baseline_value=1.0,
                current_value=1.2,
                diff_absolute=0.2,
                diff_percent=20.0,
                is_regression=False,
                threshold=20.0,
            )
        ]

        json_str = format_comparison_json(results)
        data = json.loads(json_str)

        assert "comparisons" in data
        assert "summary" in data
        assert len(data["comparisons"]) == 1


class TestSaveReport:
    """Tests for save_report."""

    def test_save_text(self):
        """Test saving text report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report"

            result = save_report("Test content", filepath, format="text")

            assert result.exists()
            assert result.suffix == ".txt"
            assert result.read_text() == "Test content"

    def test_save_json(self):
        """Test saving JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report"

            result = save_report('{"key": "value"}', filepath, format="json")

            assert result.exists()
            assert result.suffix == ".json"

    def test_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "report.txt"

            result = save_report("content", filepath)

            assert result.exists()


class TestProfileReport:
    """Tests for ProfileReport builder."""

    def test_add_profile(self):
        """Test adding profiles."""
        report = ProfileReport()
        result = ProfileResult(profile_type="cpu", wall_time=1.0)

        report.add_profile("test", result)

        assert "test" in report._profiles

    def test_chaining(self):
        """Test method chaining."""
        result1 = ProfileResult(profile_type="cpu", wall_time=1.0)
        result2 = ProfileResult(profile_type="memory", wall_time=2.0)

        report = (
            ProfileReport()
            .add_profile("cpu", result1)
            .add_profile("memory", result2)
            .set_metadata("version", "1.0")
        )

        assert len(report._profiles) == 2
        assert report._metadata["version"] == "1.0"

    def test_to_text(self):
        """Test text generation."""
        result = ProfileResult(profile_type="cpu", wall_time=1.5)
        report = ProfileReport().add_profile("analyze", result)

        text = report.to_text()

        assert "analyze" in text
        assert "1.5" in text

    def test_to_json(self):
        """Test JSON generation."""
        result = ProfileResult(profile_type="cpu", wall_time=1.5)
        report = ProfileReport().add_profile("analyze", result)

        json_str = report.to_json()
        data = json.loads(json_str)

        assert "profiles" in data
        assert "analyze" in data["profiles"]

    def test_has_regressions(self):
        """Test regression detection."""
        report = ProfileReport()

        assert not report.has_regressions()

        report.set_comparison(
            [
                ComparisonResult(
                    profile_name="test",
                    metric="wall_time",
                    baseline_value=1.0,
                    current_value=1.5,
                    diff_absolute=0.5,
                    diff_percent=50.0,
                    is_regression=True,
                    threshold=20.0,
                )
            ]
        )

        assert report.has_regressions()

    def test_save(self):
        """Test saving report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ProfileResult(profile_type="cpu", wall_time=1.0)
            report = ProfileReport().add_profile("test", result)

            path = report.save(Path(tmpdir) / "report.txt")

            assert path.exists()
            content = path.read_text()
            assert "test" in content
