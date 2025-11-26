"""Shared utilities for accuracy evaluation."""

import json
import random
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Supported audio file extensions
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"}


def discover_audio_files(source_path: Path) -> list[Path]:
    """Discover all audio files recursively in source directory.

    Args:
        source_path: Directory to search for audio files

    Returns:
        Sorted list of audio file paths
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")

    audio_files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(source_path.rglob(f"*{ext}"))

    # Sort for reproducibility
    audio_files.sort()

    logger.info("discovered audio files", count=len(audio_files), source=str(source_path))
    return audio_files


def sample_random(files: list[Path], size: int, seed: int | None = None) -> list[Path]:
    """Sample random subset of files with optional seed for reproducibility.

    Args:
        files: List of file paths
        size: Number of files to sample
        seed: Random seed for reproducibility

    Returns:
        Random sample of files
    """
    if seed is not None:
        random.seed(seed)

    sample_size = min(size, len(files))
    sampled = random.sample(files, sample_size)

    logger.info(
        "sampled files",
        strategy="random",
        sample_size=sample_size,
        total_files=len(files),
        seed=seed,
    )

    return sampled


def sample_full(files: list[Path]) -> list[Path]:
    """Return all files (no sampling)."""
    logger.info("sampled files", strategy="full", total_files=len(files))
    return files


def calculate_mae(errors: list[float]) -> float:
    """Calculate Mean Absolute Error."""
    if not errors:
        return 0.0
    return sum(abs(e) for e in errors) / len(errors)


def calculate_rmse(errors: list[float]) -> float:
    """Calculate Root Mean Square Error."""
    if not errors:
        return 0.0
    return (sum(e**2 for e in errors) / len(errors)) ** 0.5


def calculate_accuracy_within_tolerance(errors: list[float], tolerance: float) -> float:
    """Calculate percentage of values within tolerance.

    Args:
        errors: List of error values
        tolerance: Tolerance threshold

    Returns:
        Percentage (0-100) of values within tolerance
    """
    if not errors:
        return 0.0

    within_tolerance = sum(1 for e in errors if abs(e) <= tolerance)
    return (within_tolerance / len(errors)) * 100.0


def calculate_error_distribution(errors: list[float]) -> dict[str, int]:
    """Calculate error distribution histogram.

    Args:
        errors: List of error values

    Returns:
        Dictionary mapping bin range to count
    """
    if not errors:
        return {}

    # Simple binning strategy
    distribution = {
        "[-10, -5)": 0,
        "[-5, 0)": 0,
        "[0, 5)": 0,
        "[5, 10)": 0,
        "[10+)": 0,
    }

    for error in errors:
        if error < -10:
            distribution["[-10, -5)"] += 1
        elif -10 <= error < -5:
            distribution["[-10, -5)"] += 1
        elif -5 <= error < 0:
            distribution["[-5, 0)"] += 1
        elif 0 <= error < 5:
            distribution["[0, 5)"] += 1
        elif 5 <= error < 10:
            distribution["[5, 10)"] += 1
        else:
            distribution["[10+)"] += 1

    return distribution


def identify_outliers(results: list[dict], n: int = 10) -> list[dict]:
    """Identify worst N outliers by absolute error.

    Args:
        results: List of evaluation results
        n: Number of outliers to return

    Returns:
        List of worst outlier results
    """
    # Sort by absolute error descending
    sorted_results = sorted(
        [r for r in results if r.get("success", False)],
        key=lambda r: abs(r.get("error", 0)),
        reverse=True,
    )

    return sorted_results[:n]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("git commit unavailable")
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("git branch unavailable")
        return "unknown"


def save_results_json(results: dict, output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("saved results json", path=str(output_path))


def save_results_markdown(results: dict, output_path: Path) -> None:
    """Save results summary to Markdown file.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = results["metadata"]
    summary = results["summary"]
    outliers = results.get("outliers", [])

    lines = [
        f"# {metadata['analysis_type'].upper()} Evaluation Results",
        "",
        f"**Date**: {metadata['timestamp']}",
        f"**Commit**: {metadata['git_commit']}",
        f"**Sample**: {metadata['sample_size']} files ({metadata['sampling_strategy']}",
    ]

    if metadata.get("sampling_seed"):
        lines[-1] += f", seed={metadata['sampling_seed']}"
    lines[-1] += ")"

    lines.extend(
        [
            "",
            "## Summary Metrics",
            f"- MAE: {summary['mean_absolute_error']:.2f}",
            f"- RMSE: {summary['root_mean_square_error']:.2f}",
            f"- Accuracy (±{metadata['tolerance']}): {summary['accuracy_within_tolerance']:.1f}%",
            f"- Successful: {summary['successful']} / {summary['total_files']}",
            f"- Failed: {summary['failed']}",
            "",
        ]
    )

    if outliers:
        lines.extend(
            [
                "## Worst Outliers",
                "| File | Reference | Computed | Error |",
                "|------|-----------|----------|-------|",
            ]
        )

        for outlier in outliers[:10]:
            file_name = Path(outlier["file"]).name
            lines.append(
                f"| {file_name} | {outlier['reference']:.1f} | "
                f"{outlier['computed']:.1f} | {outlier['error']:.1f} |"
            )

        lines.append("")

    # Error distribution
    if "error_distribution" in summary:
        lines.extend(["## Error Distribution", ""])
        for bin_range, count in summary["error_distribution"].items():
            lines.append(f"- {bin_range}: {count} files")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("saved results markdown", path=str(output_path))


def save_error_distribution_plot(
    errors: list[float], output_path: Path, analysis_type: str = "BPM"
) -> None:
    """Save error distribution plot (optional matplotlib visualization).

    Args:
        errors: List of error values
        output_path: Output file path for plot
        analysis_type: Type of analysis for labeling
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("matplotlib unavailable", msg="skipping plot generation")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel(f"Error ({analysis_type})")
    plt.ylabel("Frequency")
    plt.title(f"{analysis_type} Error Distribution")
    plt.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("saved error plot", path=str(output_path))


def create_symlinks(output_path: Path) -> None:
    """Create 'latest' symlinks for most recent results.

    Args:
        output_path: Path to the timestamped result file (without extension)
    """
    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    png_path = output_path.with_suffix(".png")

    latest_json = output_path.parent / "latest.json"
    latest_md = output_path.parent / "latest.md"
    latest_png = output_path.parent / "latest.png"

    # Remove old symlinks if they exist
    for symlink in [latest_json, latest_md, latest_png]:
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()

    # Create new symlinks
    if json_path.exists():
        latest_json.symlink_to(json_path.name)
        logger.debug("created symlink", link=str(latest_json), target=json_path.name)

    if md_path.exists():
        latest_md.symlink_to(md_path.name)
        logger.debug("created symlink", link=str(latest_md), target=md_path.name)

    if png_path.exists():
        latest_png.symlink_to(png_path.name)
        logger.debug("created symlink", link=str(latest_png), target=png_path.name)
