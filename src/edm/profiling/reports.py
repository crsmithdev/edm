"""Profile report generation."""

import json
from pathlib import Path
from typing import Any

from edm.profiling.base import ProfileResult
from edm.profiling.baseline import Baseline, ComparisonResult


def format_profile_text(result: ProfileResult, title: str = "Profile Results") -> str:
    """Format a profile result as human-readable text.

    Args:
        result: Profile result to format.
        title: Title for the report.

    Returns:
        Formatted text report.
    """
    lines = [
        f"{'=' * 60}",
        f" {title}",
        f"{'=' * 60}",
        "",
        f"Type: {result.profile_type}",
        f"Wall Time: {result.wall_time:.3f}s",
    ]

    if result.cpu_time > 0:
        lines.append(f"CPU Time: {result.cpu_time:.3f}s")

    if result.peak_memory_mb > 0:
        lines.append(f"Peak Memory: {result.peak_memory_mb:.1f} MB")

    if result.function_stats:
        lines.extend(
            [
                "",
                "Top Functions:",
                f"{'-' * 60}",
            ]
        )

        for func in result.top_functions(10):
            lines.append(
                f"  {func.name}: {func.cumulative_time:.3f}s "
                f"({func.calls} calls, {func.total_time:.3f}s self)"
            )

    lines.append("")
    return "\n".join(lines)


def format_comparison_text(
    results: list[ComparisonResult],
    baseline_name: str = "baseline",
    current_name: str = "current",
) -> str:
    """Format comparison results as human-readable text.

    Args:
        results: List of comparison results.
        baseline_name: Name of baseline for display.
        current_name: Name of current run for display.

    Returns:
        Formatted text report.
    """
    lines = [
        f"{'=' * 70}",
        f" Performance Comparison: {baseline_name} → {current_name}",
        f"{'=' * 70}",
        "",
    ]

    regressions = [r for r in results if r.is_regression]
    improvements = [r for r in results if r.diff_percent < -10]

    if regressions:
        lines.extend(
            [
                "REGRESSIONS DETECTED:",
                "-" * 70,
            ]
        )
        for r in regressions:
            lines.append(
                f"  [{r.profile_name}] {r.metric}: "
                f"{r.baseline_value:.3f} → {r.current_value:.3f} "
                f"({r.diff_percent:+.1f}%)"
            )
        lines.append("")

    if improvements:
        lines.extend(
            [
                "Improvements:",
                "-" * 70,
            ]
        )
        for r in improvements:
            lines.append(
                f"  [{r.profile_name}] {r.metric}: "
                f"{r.baseline_value:.3f} → {r.current_value:.3f} "
                f"({r.diff_percent:+.1f}%)"
            )
        lines.append("")

    # Summary table
    lines.extend(
        [
            "All Metrics:",
            "-" * 70,
            f"{'Profile':<20} {'Metric':<15} {'Baseline':>10} {'Current':>10} {'Diff':>10}",
            "-" * 70,
        ]
    )

    for r in results:
        status = "!!!" if r.is_regression else ""
        lines.append(
            f"{r.profile_name:<20} {r.metric:<15} "
            f"{r.baseline_value:>10.3f} {r.current_value:>10.3f} "
            f"{r.diff_percent:>+9.1f}% {status}"
        )

    lines.extend(
        [
            "-" * 70,
            "",
            f"Total comparisons: {len(results)}",
            f"Regressions: {len(regressions)}",
            f"Improvements: {len(improvements)}",
            "",
        ]
    )

    return "\n".join(lines)


def format_profile_json(result: ProfileResult) -> str:
    """Format a profile result as JSON.

    Args:
        result: Profile result to format.

    Returns:
        JSON string.
    """
    return json.dumps(result.to_dict(), indent=2)


def format_baseline_json(baseline: Baseline) -> str:
    """Format a baseline as JSON.

    Args:
        baseline: Baseline to format.

    Returns:
        JSON string.
    """
    return json.dumps(baseline.to_dict(), indent=2)


def format_comparison_json(results: list[ComparisonResult]) -> str:
    """Format comparison results as JSON.

    Args:
        results: List of comparison results.

    Returns:
        JSON string.
    """
    data = {
        "comparisons": [
            {
                "profile_name": r.profile_name,
                "metric": r.metric,
                "baseline_value": r.baseline_value,
                "current_value": r.current_value,
                "diff_absolute": r.diff_absolute,
                "diff_percent": r.diff_percent,
                "is_regression": r.is_regression,
                "threshold": r.threshold,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "regressions": sum(1 for r in results if r.is_regression),
            "improvements": sum(1 for r in results if r.diff_percent < -10),
        },
    }
    return json.dumps(data, indent=2)


def save_report(
    content: str,
    filepath: Path,
    format: str = "text",
) -> Path:
    """Save a report to file.

    Args:
        content: Report content.
        filepath: Output file path.
        format: Report format (text, json).

    Returns:
        Path to saved file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Add appropriate extension if not present
    if format == "json" and not filepath.suffix:
        filepath = filepath.with_suffix(".json")
    elif format == "text" and not filepath.suffix:
        filepath = filepath.with_suffix(".txt")

    with open(filepath, "w") as f:
        f.write(content)

    return filepath


class ProfileReport:
    """Builder for profile reports.

    Example:
        report = (
            ProfileReport()
            .add_profile("analyze", cpu_result)
            .add_profile("load", memory_result)
            .set_comparison(comparison_results)
            .build()
        )
        print(report.to_text())
        report.save(Path("report.txt"))
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ProfileResult] = {}
        self._comparisons: list[ComparisonResult] = []
        self._metadata: dict[str, Any] = {}

    def add_profile(self, name: str, result: ProfileResult) -> "ProfileReport":
        """Add a profile result to the report."""
        self._profiles[name] = result
        return self

    def set_comparison(self, results: list[ComparisonResult]) -> "ProfileReport":
        """Set comparison results."""
        self._comparisons = results
        return self

    def set_metadata(self, key: str, value: Any) -> "ProfileReport":
        """Set metadata for the report."""
        self._metadata[key] = value
        return self

    def to_text(self) -> str:
        """Generate text report."""
        sections = []

        for name, result in self._profiles.items():
            sections.append(format_profile_text(result, f"Profile: {name}"))

        if self._comparisons:
            sections.append(format_comparison_text(self._comparisons))

        return "\n".join(sections)

    def to_json(self) -> str:
        """Generate JSON report."""
        data: dict[str, Any] = {
            "profiles": {name: r.to_dict() for name, r in self._profiles.items()},
            "metadata": self._metadata,
        }

        if self._comparisons:
            data["comparisons"] = [
                {
                    "profile_name": r.profile_name,
                    "metric": r.metric,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "diff_percent": r.diff_percent,
                    "is_regression": r.is_regression,
                }
                for r in self._comparisons
            ]

        return json.dumps(data, indent=2)

    def save(self, filepath: Path, format: str = "text") -> Path:
        """Save report to file."""
        content = self.to_json() if format == "json" else self.to_text()
        return save_report(content, filepath, format)

    def has_regressions(self) -> bool:
        """Check if any comparisons show regressions."""
        return any(r.is_regression for r in self._comparisons)
