"""Tests for baseline storage and comparison."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from edm.profiling.base import ProfileResult
from edm.profiling.baseline import (
    Baseline,
    BaselineMetadata,
    BaselineStore,
    compare_baseline,
    create_baseline,
    get_git_info,
    get_system_info,
    load_and_compare,
    save_baseline,
)


class TestBaselineMetadata:
    """Tests for BaselineMetadata dataclass."""

    def test_creation(self):
        """Test basic creation."""
        meta = BaselineMetadata(
            name="test",
            commit="abc123",
            branch="main",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            system_info={"python": "3.12"},
        )
        assert meta.name == "test"
        assert meta.commit == "abc123"
        assert meta.branch == "main"


class TestBaseline:
    """Tests for Baseline dataclass."""

    def test_creation(self):
        """Test basic creation."""
        baseline = Baseline(
            metadata=BaselineMetadata(
                name="test",
                commit="abc123",
                branch="main",
                timestamp=datetime.now(),
            ),
            profiles={
                "analyze_bpm": ProfileResult(
                    profile_type="cpu",
                    wall_time=2.5,
                    cpu_time=2.0,
                )
            },
        )
        assert baseline.metadata.name == "test"
        assert "analyze_bpm" in baseline.profiles

    def test_to_dict(self):
        """Test serialization to dict."""
        baseline = Baseline(
            metadata=BaselineMetadata(
                name="test",
                commit="abc123",
                branch="main",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
            ),
            profiles={
                "test_profile": ProfileResult(
                    profile_type="cpu",
                    wall_time=1.0,
                )
            },
        )

        data = baseline.to_dict()

        assert data["metadata"]["name"] == "test"
        assert data["metadata"]["commit"] == "abc123"
        assert "test_profile" in data["profiles"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "metadata": {
                "name": "main",
                "commit": "def456",
                "branch": "develop",
                "timestamp": "2025-01-01T14:00:00",
                "system_info": {"os": "linux"},
            },
            "profiles": {
                "bpm": {
                    "profile_type": "cpu",
                    "wall_time": 3.5,
                    "cpu_time": 3.0,
                    "peak_memory_mb": 0,
                    "function_stats": [],
                    "timestamp": "2025-01-01T14:00:00",
                    "metadata": {},
                }
            },
        }

        baseline = Baseline.from_dict(data)

        assert baseline.metadata.name == "main"
        assert baseline.metadata.commit == "def456"
        assert baseline.profiles["bpm"].wall_time == 3.5

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = Baseline(
            metadata=BaselineMetadata(
                name="roundtrip",
                commit="xyz",
                branch="test",
                timestamp=datetime(2025, 6, 15, 10, 30, 0),
            ),
            profiles={
                "p1": ProfileResult(profile_type="cpu", wall_time=1.0),
                "p2": ProfileResult(profile_type="memory", wall_time=2.0, peak_memory_mb=100),
            },
        )

        restored = Baseline.from_dict(original.to_dict())

        assert restored.metadata.name == original.metadata.name
        assert len(restored.profiles) == 2
        assert restored.profiles["p1"].wall_time == 1.0
        assert restored.profiles["p2"].peak_memory_mb == 100


class TestBaselineStore:
    """Tests for BaselineStore class."""

    def test_save_and_load(self):
        """Test saving and loading a baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))

            baseline = Baseline(
                metadata=BaselineMetadata(
                    name="test",
                    commit="abc",
                    branch="main",
                    timestamp=datetime.now(),
                ),
                profiles={"p1": ProfileResult(profile_type="cpu", wall_time=1.0)},
            )

            # Save
            path = store.save("test", baseline)
            assert path.exists()
            assert path.name == "test.json"

            # Load
            loaded = store.load("test")
            assert loaded is not None
            assert loaded.metadata.name == "test"
            assert loaded.profiles["p1"].wall_time == 1.0

    def test_load_nonexistent(self):
        """Test loading a nonexistent baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))
            result = store.load("nonexistent")
            assert result is None

    def test_list(self):
        """Test listing baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))

            # Empty initially
            assert store.list() == []

            # Add some baselines
            for name in ["main", "develop", "feature"]:
                baseline = Baseline(
                    metadata=BaselineMetadata(
                        name=name,
                        commit="x",
                        branch=name,
                        timestamp=datetime.now(),
                    ),
                    profiles={},
                )
                store.save(name, baseline)

            baselines = store.list()
            assert len(baselines) == 3
            assert set(baselines) == {"main", "develop", "feature"}

    def test_delete(self):
        """Test deleting a baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir))

            baseline = Baseline(
                metadata=BaselineMetadata(
                    name="delete-me",
                    commit="x",
                    branch="main",
                    timestamp=datetime.now(),
                ),
                profiles={},
            )
            store.save("delete-me", baseline)

            assert store.load("delete-me") is not None
            assert store.delete("delete-me") is True
            assert store.load("delete-me") is None
            assert store.delete("delete-me") is False  # Already deleted


class TestCompareBaseline:
    """Tests for compare_baseline function."""

    def test_wall_time_regression(self):
        """Test detecting wall time regression."""
        baseline = Baseline(
            metadata=BaselineMetadata(
                name="test",
                commit="x",
                branch="main",
                timestamp=datetime.now(),
            ),
            profiles={
                "test": ProfileResult(profile_type="cpu", wall_time=1.0),
            },
        )

        current = {
            "test": ProfileResult(profile_type="cpu", wall_time=1.5),  # 50% slower
        }

        results = compare_baseline(baseline, current)

        assert len(results) == 1
        assert results[0].profile_name == "test"
        assert results[0].metric == "wall_time"
        assert results[0].baseline_value == 1.0
        assert results[0].current_value == 1.5
        assert results[0].diff_percent == 50.0
        assert results[0].is_regression is True

    def test_no_regression(self):
        """Test no regression when within threshold."""
        baseline = Baseline(
            metadata=BaselineMetadata(
                name="test",
                commit="x",
                branch="main",
                timestamp=datetime.now(),
            ),
            profiles={
                "test": ProfileResult(profile_type="cpu", wall_time=1.0),
            },
        )

        current = {
            "test": ProfileResult(profile_type="cpu", wall_time=1.1),  # 10% slower
        }

        results = compare_baseline(baseline, current)

        assert len(results) == 1
        assert results[0].is_regression is False

    def test_multiple_metrics(self):
        """Test comparing multiple metrics."""
        baseline = Baseline(
            metadata=BaselineMetadata(
                name="test",
                commit="x",
                branch="main",
                timestamp=datetime.now(),
            ),
            profiles={
                "test": ProfileResult(
                    profile_type="cpu",
                    wall_time=1.0,
                    cpu_time=0.8,
                    peak_memory_mb=100,
                ),
            },
        )

        current = {
            "test": ProfileResult(
                profile_type="cpu",
                wall_time=1.0,
                cpu_time=0.8,
                peak_memory_mb=100,
            ),
        }

        results = compare_baseline(baseline, current)

        # Should compare wall_time, cpu_time, and peak_memory
        assert len(results) == 3
        metrics = {r.metric for r in results}
        assert metrics == {"wall_time", "cpu_time", "peak_memory_mb"}


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_git_info(self):
        """Test getting git info."""
        commit, branch = get_git_info()
        # Should return something (even if "unknown")
        assert isinstance(commit, str)
        assert isinstance(branch, str)
        assert len(commit) > 0
        assert len(branch) > 0

    def test_get_system_info(self):
        """Test getting system info."""
        info = get_system_info()
        assert "python_version" in info
        assert "platform" in info
        assert "machine" in info

    def test_create_baseline(self):
        """Test creating a baseline with auto-populated metadata."""
        profiles = {
            "test": ProfileResult(profile_type="cpu", wall_time=1.0),
        }

        baseline = create_baseline("my-baseline", profiles)

        assert baseline.metadata.name == "my-baseline"
        assert baseline.metadata.commit != ""
        assert baseline.metadata.branch != ""
        assert "python_version" in baseline.metadata.system_info
        assert baseline.profiles["test"].wall_time == 1.0

    def test_save_baseline_convenience(self):
        """Test save_baseline convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles = {
                "test": ProfileResult(profile_type="cpu", wall_time=1.0),
            }

            path = save_baseline("test", profiles, Path(tmpdir))

            assert path.exists()
            assert path.name == "test.json"

    def test_load_and_compare(self):
        """Test load_and_compare convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a baseline first
            baseline_profiles = {
                "test": ProfileResult(profile_type="cpu", wall_time=1.0),
            }
            save_baseline("main", baseline_profiles, Path(tmpdir))

            # Compare against it
            current = {
                "test": ProfileResult(profile_type="cpu", wall_time=1.3),
            }

            results = load_and_compare("main", current, Path(tmpdir))

            assert results is not None
            assert len(results) == 1
            assert results[0].diff_percent == pytest.approx(30.0)

    def test_load_and_compare_missing_baseline(self):
        """Test load_and_compare with missing baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current = {"test": ProfileResult(profile_type="cpu", wall_time=1.0)}
            results = load_and_compare("nonexistent", current, Path(tmpdir))
            assert results is None
