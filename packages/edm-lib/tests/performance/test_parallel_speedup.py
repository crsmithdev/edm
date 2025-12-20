"""Performance tests for parallel processing speedup.

These tests measure wall-clock time for different worker counts and batch sizes.
They are marked to skip in CI as they take significant time to run.
"""

import json
import time
from pathlib import Path

import pytest
from edm_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

# Skip these tests in CI - they're too slow
pytestmark = pytest.mark.skip(reason="Performance tests - run manually only")


@pytest.fixture
def test_audio_files():
    """Get list of test audio files."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    files = [
        fixtures_dir / "beat_120bpm.wav",
        fixtures_dir / "beat_125bpm.wav",
        fixtures_dir / "beat_128bpm.wav",
        fixtures_dir / "beat_140bpm.wav",
        fixtures_dir / "beat_150bpm.wav",
        fixtures_dir / "beat_174bpm.wav",
        fixtures_dir / "click_120bpm.wav",
        fixtures_dir / "click_125bpm.wav",
        fixtures_dir / "click_128bpm.wav",
        fixtures_dir / "click_140bpm.wav",
    ]
    return [f for f in files if f.exists()]


def measure_analyze_time(files, workers, tmp_path):
    """Measure time to analyze files with given worker count."""
    output_file = tmp_path / f"results_w{workers}.json"
    file_args = [str(f) for f in files]

    start = time.time()
    result = runner.invoke(
        app,
        ["analyze", "--workers", str(workers), "--output", str(output_file), "--no-color"]
        + file_args,
    )
    elapsed = time.time() - start

    assert result.exit_code == 0
    return elapsed


def test_parallel_speedup_10_files(test_audio_files, tmp_path):
    """Measure speedup for 10 files with different worker counts."""
    files = test_audio_files[:10]
    timings = {}

    for workers in [1, 2, 4, 8]:
        elapsed = measure_analyze_time(files, workers, tmp_path)
        timings[workers] = elapsed
        print(f"Workers={workers}: {elapsed:.2f}s")

    # Calculate speedups
    baseline = timings[1]
    print("\nSpeedups relative to workers=1:")
    for workers, elapsed in timings.items():
        speedup = baseline / elapsed
        print(f"  Workers={workers}: {speedup:.2f}x ({elapsed:.2f}s)")

    # Verify we get some speedup with more workers
    assert timings[4] < timings[1], "4 workers should be faster than 1"


def test_parallel_speedup_20_files(test_audio_files, tmp_path):
    """Measure speedup for 20 files with different worker counts."""
    # Duplicate files to get 20 total
    files = (test_audio_files * 2)[:20]
    timings = {}

    for workers in [1, 2, 4, 8]:
        elapsed = measure_analyze_time(files, workers, tmp_path)
        timings[workers] = elapsed
        print(f"Workers={workers}: {elapsed:.2f}s")

    # Calculate speedups
    baseline = timings[1]
    print("\nSpeedups relative to workers=1:")
    for workers, elapsed in timings.items():
        speedup = baseline / elapsed
        print(f"  Workers={workers}: {speedup:.2f}x ({elapsed:.2f}s)")

    # Verify we get better speedup with more files
    assert timings[8] < timings[1], "8 workers should be faster than 1"


def test_parallel_efficiency(test_audio_files, tmp_path):
    """Test parallel efficiency (speedup / workers)."""
    files = (test_audio_files * 2)[:20]
    timings = {}

    for workers in [1, 2, 4, 8]:
        elapsed = measure_analyze_time(files, workers, tmp_path)
        timings[workers] = elapsed

    baseline = timings[1]
    print("\nParallel efficiency:")
    for workers, elapsed in timings.items():
        if workers > 1:
            speedup = baseline / elapsed
            efficiency = (speedup / workers) * 100
            print(f"  Workers={workers}: {efficiency:.1f}% efficient")

    # With CPU-bound work, efficiency should be reasonable
    speedup_4 = baseline / timings[4]
    efficiency_4 = (speedup_4 / 4) * 100
    # Should get at least 50% efficiency with 4 workers
    assert efficiency_4 > 50, f"4-worker efficiency too low: {efficiency_4:.1f}%"


def test_generate_speedup_report(test_audio_files, tmp_path):
    """Generate comprehensive speedup report."""
    batch_sizes = [5, 10, 20]
    worker_counts = [1, 2, 4, 8]

    report = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_sizes": {},
    }

    for batch_size in batch_sizes:
        files = (test_audio_files * 10)[:batch_size]
        batch_report = {"timings": {}, "speedups": {}}

        print(f"\n=== Batch size: {batch_size} files ===")

        for workers in worker_counts:
            elapsed = measure_analyze_time(files, workers, tmp_path)
            batch_report["timings"][workers] = elapsed
            print(f"  Workers={workers}: {elapsed:.2f}s")

        baseline = batch_report["timings"][1]
        print("  Speedups:")
        for workers, elapsed in batch_report["timings"].items():
            speedup = baseline / elapsed
            batch_report["speedups"][workers] = speedup
            print(f"    Workers={workers}: {speedup:.2f}x")

        report["batch_sizes"][batch_size] = batch_report

    # Save report
    report_file = tmp_path / "speedup_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")


@pytest.mark.manual
def test_memory_usage(test_audio_files, tmp_path):
    """Test memory usage with different worker counts.

    This is a manual test - monitor memory with system tools.
    """
    files = (test_audio_files * 10)[:50]

    for workers in [1, 4, 8]:
        print(f"\nAnalyzing with workers={workers}")
        print("Monitor memory usage now...")
        input("Press Enter to start...")

        output_file = tmp_path / f"results_w{workers}_memory.json"
        file_args = [str(f) for f in files]

        result = runner.invoke(
            app,
            ["analyze", "--workers", str(workers), "--output", str(output_file), "--no-color"]
            + file_args,
        )

        assert result.exit_code == 0
        print("Done")
        input("Press Enter to continue to next test...")
