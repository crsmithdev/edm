"""Integration tests for parallel analyze command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@pytest.fixture
def test_audio_files():
    """Get list of test audio files."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return [
        fixtures_dir / "beat_120bpm.wav",
        fixtures_dir / "beat_125bpm.wav",
        fixtures_dir / "beat_128bpm.wav",
        fixtures_dir / "beat_140bpm.wav",
        fixtures_dir / "beat_150bpm.wav",
    ]


def test_analyze_parallel_workers_1(test_audio_files, tmp_path):
    """Test analyze with workers=1."""
    output_file = tmp_path / "results.json"
    files = [str(f) for f in test_audio_files[:3]]

    result = runner.invoke(
        app,
        ["analyze", "--workers", "1", "--output", str(output_file), "--no-color"] + files,
    )

    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        results = json.load(f)

    assert len(results) == 3
    for res in results:
        assert "bpm" in res
        assert "file" in res


def test_analyze_parallel_workers_4(test_audio_files, tmp_path):
    """Test analyze with workers=4."""
    output_file = tmp_path / "results.json"
    files = [str(f) for f in test_audio_files]

    result = runner.invoke(
        app,
        ["analyze", "--workers", "4", "--output", str(output_file), "--no-color"] + files,
    )

    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        results = json.load(f)

    assert len(results) == 5
    for res in results:
        assert "bpm" in res
        assert "file" in res


def test_analyze_parallel_deterministic(test_audio_files, tmp_path):
    """Test that parallel processing produces deterministic results."""
    output_1 = tmp_path / "results_1.json"
    output_2 = tmp_path / "results_2.json"
    files = [str(f) for f in test_audio_files[:3]]

    # Run twice with workers=4
    result_1 = runner.invoke(
        app,
        ["analyze", "--workers", "4", "--output", str(output_1), "--no-color"] + files,
    )
    result_2 = runner.invoke(
        app,
        ["analyze", "--workers", "4", "--output", str(output_2), "--no-color"] + files,
    )

    assert result_1.exit_code == 0
    assert result_2.exit_code == 0

    with open(output_1) as f:
        results_1 = json.load(f)
    with open(output_2) as f:
        results_2 = json.load(f)

    # Sort by file to ensure order
    results_1.sort(key=lambda x: x["file"])
    results_2.sort(key=lambda x: x["file"])

    # BPM values should be identical
    for r1, r2 in zip(results_1, results_2):
        assert r1["file"] == r2["file"]
        assert r1["bpm"] == r2["bpm"]
        assert r1["bpm_source"] == r2["bpm_source"]


def test_analyze_parallel_error_handling(tmp_path):
    """Test error handling with invalid file."""
    output_file = tmp_path / "results.json"
    invalid_file = tmp_path / "nonexistent.mp3"

    result = runner.invoke(
        app,
        ["analyze", "--workers", "2", "--output", str(output_file), str(invalid_file)],
    )

    # CLI validation rejects nonexistent files before processing
    # Exit code 2 indicates CLI error (file not found)
    assert result.exit_code == 2


def test_analyze_parallel_different_worker_counts(test_audio_files, tmp_path):
    """Test that different worker counts produce same BPM results."""
    files = [str(f) for f in test_audio_files[:5]]

    outputs = {}
    for workers in [1, 2, 4]:
        output_file = tmp_path / f"results_w{workers}.json"
        result = runner.invoke(
            app,
            ["analyze", "--workers", str(workers), "--output", str(output_file), "--no-color"]
            + files,
        )
        assert result.exit_code == 0

        with open(output_file) as f:
            results = json.load(f)
        results.sort(key=lambda x: x["file"])
        outputs[workers] = results

    # All worker counts should produce identical BPM values
    for i in range(len(files)):
        bpm_1 = outputs[1][i]["bpm"]
        bpm_2 = outputs[2][i]["bpm"]
        bpm_4 = outputs[4][i]["bpm"]
        assert bpm_1 == bpm_2 == bpm_4


def test_analyze_parallel_batch_sizes(test_audio_files, tmp_path):
    """Test different batch sizes."""
    test_cases = [
        (test_audio_files[:1], 1),  # Single file
        (test_audio_files[:3], 3),  # Small batch
        (test_audio_files[:5], 5),  # Larger batch
    ]

    for files, expected_count in test_cases:
        output_file = tmp_path / f"results_{expected_count}.json"
        file_args = [str(f) for f in files]

        result = runner.invoke(
            app,
            ["analyze", "--workers", "4", "--output", str(output_file), "--no-color"] + file_args,
        )

        assert result.exit_code == 0

        with open(output_file) as f:
            results = json.load(f)

        assert len(results) == expected_count
