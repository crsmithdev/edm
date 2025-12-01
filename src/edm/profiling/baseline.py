"""Baseline storage and comparison for performance regression detection."""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from edm.profiling.base import ProfileResult

logger = structlog.get_logger(__name__)

# Default threshold for regression detection (20% slower)
DEFAULT_REGRESSION_THRESHOLD = 0.20


@dataclass
class BaselineMetadata:
    """Metadata for a baseline.

    Attributes:
        name: Baseline name/identifier.
        commit: Git commit hash when baseline was created.
        branch: Git branch name.
        timestamp: When the baseline was created.
        system_info: System information (Python version, OS, etc.).
    """

    name: str
    commit: str
    branch: str
    timestamp: datetime
    system_info: dict[str, str] = field(default_factory=dict)


@dataclass
class Baseline:
    """A performance baseline containing profile results.

    Attributes:
        metadata: Baseline metadata.
        profiles: Dict mapping profile name to ProfileResult.
    """

    metadata: BaselineMetadata
    profiles: dict[str, ProfileResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "commit": self.metadata.commit,
                "branch": self.metadata.branch,
                "timestamp": self.metadata.timestamp.isoformat(),
                "system_info": self.metadata.system_info,
            },
            "profiles": {name: result.to_dict() for name, result in self.profiles.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        """Create from dictionary."""
        meta = data["metadata"]
        return cls(
            metadata=BaselineMetadata(
                name=meta["name"],
                commit=meta["commit"],
                branch=meta["branch"],
                timestamp=datetime.fromisoformat(meta["timestamp"]),
                system_info=meta.get("system_info", {}),
            ),
            profiles={
                name: ProfileResult.from_dict(profile)
                for name, profile in data.get("profiles", {}).items()
            },
        )


@dataclass
class ComparisonResult:
    """Result of comparing current performance against baseline.

    Attributes:
        profile_name: Name of the profile being compared.
        baseline_value: Baseline metric value.
        current_value: Current metric value.
        diff_absolute: Absolute difference.
        diff_percent: Percentage difference.
        metric: Name of the metric being compared.
        is_regression: Whether this is a regression.
        threshold: Regression threshold used.
    """

    profile_name: str
    baseline_value: float
    current_value: float
    diff_absolute: float
    diff_percent: float
    metric: str
    is_regression: bool
    threshold: float


class BaselineStore:
    """Storage for performance baselines.

    Baselines are stored as JSON files in the baselines directory.

    Example:
        store = BaselineStore(Path("benchmarks/baselines"))
        store.save("main", baseline)
        loaded = store.load("main")
    """

    def __init__(self, baselines_dir: Path | None = None) -> None:
        """Initialize baseline store.

        Args:
            baselines_dir: Directory for storing baselines.
                Defaults to benchmarks/baselines in current directory.
        """
        if baselines_dir is None:
            baselines_dir = Path.cwd() / "benchmarks" / "baselines"
        self._baselines_dir = baselines_dir

    @property
    def baselines_dir(self) -> Path:
        """Get the baselines directory."""
        return self._baselines_dir

    def save(self, name: str, baseline: Baseline) -> Path:
        """Save a baseline to disk.

        Args:
            name: Baseline name (used as filename).
            baseline: Baseline to save.

        Returns:
            Path to the saved baseline file.
        """
        self._baselines_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._baselines_dir / f"{name}.json"

        with open(filepath, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)

        logger.info("saved baseline", name=name, path=str(filepath))
        return filepath

    def load(self, name: str) -> Baseline | None:
        """Load a baseline from disk.

        Args:
            name: Baseline name.

        Returns:
            Baseline if found, None otherwise.
        """
        filepath = self._baselines_dir / f"{name}.json"

        if not filepath.exists():
            logger.debug("baseline not found", name=name, path=str(filepath))
            return None

        with open(filepath) as f:
            data = json.load(f)

        logger.debug("loaded baseline", name=name)
        return Baseline.from_dict(data)

    def list(self) -> list[str]:
        """List all available baselines.

        Returns:
            List of baseline names.
        """
        if not self._baselines_dir.exists():
            return []

        return [f.stem for f in self._baselines_dir.glob("*.json")]

    def delete(self, name: str) -> bool:
        """Delete a baseline.

        Args:
            name: Baseline name.

        Returns:
            True if deleted, False if not found.
        """
        filepath = self._baselines_dir / f"{name}.json"

        if not filepath.exists():
            return False

        filepath.unlink()
        logger.info("deleted baseline", name=name)
        return True


def get_git_info() -> tuple[str, str]:
    """Get current git commit hash and branch.

    Returns:
        Tuple of (commit_hash, branch_name). Returns ("unknown", "unknown") if git unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return commit, branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown"


def get_system_info() -> dict[str, str]:
    """Get system information for baseline metadata.

    Returns:
        Dict with Python version, platform, etc.
    """
    import platform
    import sys

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
    }


def create_baseline(
    name: str,
    profiles: dict[str, ProfileResult],
) -> Baseline:
    """Create a baseline with current git and system info.

    Args:
        name: Baseline name.
        profiles: Dict mapping profile name to ProfileResult.

    Returns:
        Baseline with metadata populated.
    """
    commit, branch = get_git_info()

    return Baseline(
        metadata=BaselineMetadata(
            name=name,
            commit=commit,
            branch=branch,
            timestamp=datetime.now(),
            system_info=get_system_info(),
        ),
        profiles=profiles,
    )


def compare_baseline(
    baseline: Baseline,
    current: dict[str, ProfileResult],
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> list[ComparisonResult]:
    """Compare current results against a baseline.

    Args:
        baseline: Baseline to compare against.
        current: Current profile results.
        threshold: Regression threshold (default 20%).

    Returns:
        List of comparison results for each metric.
    """
    results = []

    for profile_name, current_result in current.items():
        baseline_result = baseline.profiles.get(profile_name)
        if baseline_result is None:
            logger.debug("no baseline for profile", profile_name=profile_name)
            continue

        # Compare wall time
        if baseline_result.wall_time > 0:
            diff = current_result.wall_time - baseline_result.wall_time
            diff_pct = diff / baseline_result.wall_time

            results.append(
                ComparisonResult(
                    profile_name=profile_name,
                    baseline_value=baseline_result.wall_time,
                    current_value=current_result.wall_time,
                    diff_absolute=diff,
                    diff_percent=diff_pct * 100,
                    metric="wall_time",
                    is_regression=diff_pct > threshold,
                    threshold=threshold * 100,
                )
            )

        # Compare CPU time if available
        if baseline_result.cpu_time > 0 and current_result.cpu_time > 0:
            diff = current_result.cpu_time - baseline_result.cpu_time
            diff_pct = diff / baseline_result.cpu_time

            results.append(
                ComparisonResult(
                    profile_name=profile_name,
                    baseline_value=baseline_result.cpu_time,
                    current_value=current_result.cpu_time,
                    diff_absolute=diff,
                    diff_percent=diff_pct * 100,
                    metric="cpu_time",
                    is_regression=diff_pct > threshold,
                    threshold=threshold * 100,
                )
            )

        # Compare peak memory if available
        if baseline_result.peak_memory_mb > 0 and current_result.peak_memory_mb > 0:
            diff = current_result.peak_memory_mb - baseline_result.peak_memory_mb
            diff_pct = diff / baseline_result.peak_memory_mb

            results.append(
                ComparisonResult(
                    profile_name=profile_name,
                    baseline_value=baseline_result.peak_memory_mb,
                    current_value=current_result.peak_memory_mb,
                    diff_absolute=diff,
                    diff_percent=diff_pct * 100,
                    metric="peak_memory_mb",
                    is_regression=diff_pct > threshold,
                    threshold=threshold * 100,
                )
            )

    return results


def save_baseline(
    name: str,
    profiles: dict[str, ProfileResult],
    baselines_dir: Path | None = None,
) -> Path:
    """Convenience function to create and save a baseline.

    Args:
        name: Baseline name.
        profiles: Dict mapping profile name to ProfileResult.
        baselines_dir: Optional custom baselines directory.

    Returns:
        Path to saved baseline file.
    """
    baseline = create_baseline(name, profiles)
    store = BaselineStore(baselines_dir)
    return store.save(name, baseline)


def load_and_compare(
    name: str,
    current: dict[str, ProfileResult],
    baselines_dir: Path | None = None,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> list[ComparisonResult] | None:
    """Load a baseline and compare against current results.

    Args:
        name: Baseline name to load.
        current: Current profile results.
        baselines_dir: Optional custom baselines directory.
        threshold: Regression threshold (default 20%).

    Returns:
        List of comparison results, or None if baseline not found.
    """
    store = BaselineStore(baselines_dir)
    baseline = store.load(name)

    if baseline is None:
        return None

    return compare_baseline(baseline, current, threshold)
