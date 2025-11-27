"""Integration tests for parallel evaluate command."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@pytest.fixture
def test_audio_dir():
    """Get test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def reference_csv(test_audio_dir, tmp_path):
    """Create a CSV reference file for testing."""
    csv_path = tmp_path / "reference.csv"

    # Create reference data with known BPMs from test fixtures
    reference_data = [
        (test_audio_dir / "beat_120bpm.wav", 120.0),
        (test_audio_dir / "beat_125bpm.wav", 125.0),
        (test_audio_dir / "beat_128bpm.wav", 128.0),
        (test_audio_dir / "beat_140bpm.wav", 140.0),
        (test_audio_dir / "beat_150bpm.wav", 150.0),
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bpm"])  # Column must be named 'path', not 'filepath'
        for filepath, bpm in reference_data:
            writer.writerow([str(filepath.resolve()), bpm])

    return csv_path


def test_evaluate_bpm_workers_1(test_audio_dir, reference_csv, tmp_path):
    """Test evaluate bpm with workers=1."""
    output_dir = tmp_path / "eval_w1"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "bpm",
            "--source",
            str(test_audio_dir),
            "--reference",
            str(reference_csv),
            "--output",
            str(output_dir),
            "--workers",
            "1",
            "--full",
        ],
    )

    assert result.exit_code == 0

    # Check output files exist
    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) > 0

    # Verify results structure
    with open(json_files[0]) as f:
        results = json.load(f)

    assert "metadata" in results
    assert "summary" in results
    assert "mean_absolute_error" in results["summary"]
    assert "root_mean_square_error" in results["summary"]
    assert "accuracy_within_tolerance" in results["summary"]


def test_evaluate_bpm_workers_4(test_audio_dir, reference_csv, tmp_path):
    """Test evaluate bpm with workers=4."""
    output_dir = tmp_path / "eval_w4"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "bpm",
            "--source",
            str(test_audio_dir),
            "--reference",
            str(reference_csv),
            "--output",
            str(output_dir),
            "--workers",
            "4",
            "--full",
        ],
    )

    assert result.exit_code == 0

    # Check output files exist
    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) > 0

    # Verify results structure
    with open(json_files[0]) as f:
        results = json.load(f)

    assert "metadata" in results
    assert "summary" in results


def test_evaluate_bpm_different_workers_identical_metrics(test_audio_dir, reference_csv, tmp_path):
    """Test that different worker counts produce identical evaluation metrics."""
    results_by_workers = {}

    for workers in [1, 2, 4]:
        output_dir = tmp_path / f"eval_w{workers}"

        result = runner.invoke(
            app,
            [
                "evaluate",
                "bpm",
                "--source",
                str(test_audio_dir),
                "--reference",
                str(reference_csv),
                "--output",
                str(output_dir),
                "--workers",
                str(workers),
                "--full",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0

        # Load results
        json_files = list(output_dir.glob("*.json"))
        with open(json_files[0]) as f:
            results = json.load(f)

        results_by_workers[workers] = results["summary"]

    # All worker counts should produce identical metrics
    mae_1 = results_by_workers[1]["mean_absolute_error"]
    mae_2 = results_by_workers[2]["mean_absolute_error"]
    mae_4 = results_by_workers[4]["mean_absolute_error"]

    rmse_1 = results_by_workers[1]["root_mean_square_error"]
    rmse_2 = results_by_workers[2]["root_mean_square_error"]
    rmse_4 = results_by_workers[4]["root_mean_square_error"]

    accuracy_1 = results_by_workers[1]["accuracy_within_tolerance"]
    accuracy_2 = results_by_workers[2]["accuracy_within_tolerance"]
    accuracy_4 = results_by_workers[4]["accuracy_within_tolerance"]

    # Metrics should be identical
    assert mae_1 == mae_2 == mae_4
    assert rmse_1 == rmse_2 == rmse_4
    assert accuracy_1 == accuracy_2 == accuracy_4


def test_evaluate_bpm_sample_size(test_audio_dir, reference_csv, tmp_path):
    """Test evaluation with different sample sizes."""
    output_dir = tmp_path / "eval_sample"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "bpm",
            "--source",
            str(test_audio_dir),
            "--reference",
            str(reference_csv),
            "--output",
            str(output_dir),
            "--workers",
            "2",
            "--sample-size",
            "3",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0

    json_files = list(output_dir.glob("*.json"))
    with open(json_files[0]) as f:
        results = json.load(f)

    # Should have evaluated 3 files (or less if fewer files available)
    assert len(results.get("results", [])) <= 3


def test_evaluate_bpm_tolerance(test_audio_dir, reference_csv, tmp_path):
    """Test evaluation with custom tolerance."""
    output_dir = tmp_path / "eval_tolerance"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "bpm",
            "--source",
            str(test_audio_dir),
            "--reference",
            str(reference_csv),
            "--output",
            str(output_dir),
            "--workers",
            "2",
            "--full",
            "--tolerance",
            "5.0",
        ],
    )

    assert result.exit_code == 0

    json_files = list(output_dir.glob("*.json"))
    with open(json_files[0]) as f:
        results = json.load(f)

    # Tolerance should be reflected in metadata
    assert results["metadata"]["tolerance"] == 5.0
