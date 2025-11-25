"""Tests for evaluation.common module."""

import json
from pathlib import Path

import pytest

from edm.evaluation.common import (
    calculate_accuracy_within_tolerance,
    calculate_error_distribution,
    calculate_mae,
    calculate_rmse,
    discover_audio_files,
    get_git_branch,
    get_git_commit,
    identify_outliers,
    sample_full,
    sample_random,
    save_results_json,
    save_results_markdown,
)


def test_discover_audio_files(tmp_path):
    """Test discovering audio files in directory."""
    # Create test files
    (tmp_path / "track1.mp3").touch()
    (tmp_path / "track2.flac").touch()
    (tmp_path / "track3.wav").touch()
    (tmp_path / "other.txt").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "track4.m4a").touch()

    files = discover_audio_files(tmp_path)

    assert len(files) == 4
    assert all(f.suffix in {".mp3", ".flac", ".wav", ".m4a"} for f in files)
    assert (tmp_path / "other.txt") not in files


def test_discover_audio_files_nonexistent(tmp_path):
    """Test discovering files in nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        discover_audio_files(tmp_path / "nonexistent")


def test_sample_random():
    """Test random sampling with seed."""
    files = [Path(f"file{i}.mp3") for i in range(100)]

    # Same seed produces same sample
    sample1 = sample_random(files, 10, seed=42)
    sample2 = sample_random(files, 10, seed=42)
    assert sample1 == sample2

    # Different seeds produce different samples
    sample3 = sample_random(files, 10, seed=123)
    assert sample1 != sample3

    # Sample size is respected
    assert len(sample1) == 10

    # Sample size larger than files returns all files
    sample4 = sample_random(files, 200, seed=42)
    assert len(sample4) == 100


def test_sample_full():
    """Test full sampling (returns all files)."""
    files = [Path(f"file{i}.mp3") for i in range(100)]
    sampled = sample_full(files)
    assert sampled == files


def test_calculate_mae():
    """Test Mean Absolute Error calculation."""
    errors = [1.0, -2.0, 3.0, -4.0, 5.0]
    mae = calculate_mae(errors)
    assert mae == 3.0

    # Empty list
    assert calculate_mae([]) == 0.0


def test_calculate_rmse():
    """Test Root Mean Square Error calculation."""
    errors = [1.0, -1.0, 2.0, -2.0]
    rmse = calculate_rmse(errors)
    expected = (sum([1, 1, 4, 4]) / 4) ** 0.5
    assert abs(rmse - expected) < 0.001

    # Empty list
    assert calculate_rmse([]) == 0.0


def test_calculate_accuracy_within_tolerance():
    """Test accuracy within tolerance calculation."""
    errors = [0.5, -1.0, 2.0, -2.5, 3.0, -4.0]
    tolerance = 2.0

    accuracy = calculate_accuracy_within_tolerance(errors, tolerance)
    # 3 out of 6 are within ±2.0 (0.5, -1.0, 2.0)
    # -2.5 is NOT within ±2.0 (abs(-2.5) > 2.0)
    assert abs(accuracy - 50.0) < 0.1

    # Empty list
    assert calculate_accuracy_within_tolerance([], 2.0) == 0.0


def test_calculate_error_distribution():
    """Test error distribution histogram."""
    errors = [-12.0, -7.0, -3.0, -1.0, 0.5, 2.0, 3.5, 6.0, 8.0, 15.0]
    dist = calculate_error_distribution(errors)

    assert dist["[-10, -5)"] >= 1  # -12.0, -7.0
    assert dist["[-5, 0)"] >= 2  # -3.0, -1.0
    assert dist["[0, 5)"] >= 3  # 0.5, 2.0, 3.5
    assert dist["[5, 10)"] >= 2  # 6.0, 8.0
    assert dist["[10+)"] >= 1  # 15.0

    # Empty list
    assert calculate_error_distribution([]) == {}


def test_identify_outliers():
    """Test identifying worst outliers."""
    results = [
        {"file": "track1.mp3", "error": 1.0, "success": True},
        {"file": "track2.mp3", "error": -5.0, "success": True},
        {"file": "track3.mp3", "error": 10.0, "success": True},
        {"file": "track4.mp3", "error": -2.0, "success": True},
        {"file": "track5.mp3", "error": 0.5, "success": True},
        {"file": "track6.mp3", "error": None, "success": False},  # Failed
    ]

    outliers = identify_outliers(results, n=3)

    # Should return top 3 by absolute error
    assert len(outliers) == 3
    assert outliers[0]["file"] == "track3.mp3"  # abs(10.0)
    assert outliers[1]["file"] == "track2.mp3"  # abs(-5.0)
    assert outliers[2]["file"] == "track4.mp3"  # abs(-2.0)


def test_get_git_commit():
    """Test getting git commit hash."""
    commit = get_git_commit()
    # Either returns a commit hash or 'unknown'
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_get_git_branch():
    """Test getting git branch name."""
    branch = get_git_branch()
    # Either returns a branch name or 'unknown'
    assert isinstance(branch, str)
    assert len(branch) > 0


def test_save_results_json(tmp_path):
    """Test saving results to JSON file."""
    results = {
        "metadata": {"analysis_type": "bpm", "timestamp": "2025-11-24"},
        "summary": {"mae": 1.84, "rmse": 2.91},
        "results": [],
    }

    output_path = tmp_path / "results.json"
    save_results_json(results, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        loaded = json.load(f)

    assert loaded == results


def test_save_results_markdown(tmp_path):
    """Test saving results to Markdown file."""
    results = {
        "metadata": {
            "analysis_type": "bpm",
            "timestamp": "2025-11-24",
            "git_commit": "abc123",
            "sample_size": 100,
            "sampling_strategy": "random",
            "sampling_seed": 42,
            "tolerance": 2.5,
        },
        "summary": {
            "total_files": 100,
            "successful": 97,
            "failed": 3,
            "mean_absolute_error": 1.84,
            "root_mean_square_error": 2.91,
            "accuracy_within_tolerance": 87.6,
            "error_distribution": {"[0, 5)": 78},
        },
        "outliers": [
            {
                "file": "/path/to/track1.mp3",
                "reference": 128.0,
                "computed": 85.3,
                "error": -42.7,
            }
        ],
    }

    output_path = tmp_path / "results.md"
    save_results_markdown(results, output_path)

    assert output_path.exists()

    content = output_path.read_text()

    # Check key sections
    assert "# BPM Evaluation Results" in content
    assert "MAE: 1.84" in content
    assert "RMSE: 2.91" in content
    assert "Accuracy" in content
    assert "Worst Outliers" in content
    assert "track1.mp3" in content
