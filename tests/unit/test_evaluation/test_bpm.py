"""Tests for evaluation.evaluators.bpm module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from edm.evaluation.evaluators.bpm import evaluate_bpm


@patch("edm.evaluation.evaluators.bpm.analyze_bpm")
@patch("edm.evaluation.evaluators.bpm.load_reference_auto")
@patch("edm.evaluation.evaluators.bpm.discover_audio_files")
def test_evaluate_bpm_basic(mock_discover, mock_load_ref, mock_analyze, tmp_path):
    """Test basic BPM evaluation."""
    # Mock file discovery
    mock_files = [
        Path("/music/track1.mp3").resolve(),
        Path("/music/track2.flac").resolve(),
    ]
    mock_discover.return_value = mock_files

    # Mock reference data
    mock_load_ref.return_value = {
        Path("/music/track1.mp3").resolve(): 128.0,
        Path("/music/track2.flac").resolve(): 140.0,
    }

    # Mock BPM analysis
    mock_analyze.side_effect = [
        {"bpm": 127.8, "confidence": 0.95},
        {"bpm": 139.5, "confidence": 0.92},
    ]

    # Run evaluation
    output_dir = tmp_path / "results"
    results = evaluate_bpm(
        source_path=Path("/music"),
        reference_source="metadata",
        sample_size=100,
        output_dir=output_dir,
        seed=42,
        full=False,
        tolerance=2.5,
    )

    # Check results structure
    assert "metadata" in results
    assert "summary" in results
    assert "results" in results
    assert "outliers" in results

    # Check metadata
    assert results["metadata"]["analysis_type"] == "bpm"
    assert results["metadata"]["sample_size"] == 2
    assert results["metadata"]["sampling_strategy"] == "random"
    assert results["metadata"]["sampling_seed"] == 42

    # Check summary
    assert results["summary"]["total_files"] == 2
    assert results["summary"]["successful"] == 2
    assert results["summary"]["failed"] == 0
    assert "mean_absolute_error" in results["summary"]
    assert "root_mean_square_error" in results["summary"]
    assert "accuracy_within_tolerance" in results["summary"]

    # Check individual results
    assert len(results["results"]) == 2
    assert results["results"][0]["success"] is True
    assert results["results"][1]["success"] is True


@patch("edm.evaluation.evaluators.bpm.analyze_bpm")
@patch("edm.evaluation.evaluators.bpm.load_reference_auto")
@patch("edm.evaluation.evaluators.bpm.discover_audio_files")
def test_evaluate_bpm_with_failures(mock_discover, mock_load_ref, mock_analyze, tmp_path):
    """Test BPM evaluation with some failures."""
    # Mock file discovery
    mock_files = [
        Path("/music/track1.mp3").resolve(),
        Path("/music/track2.flac").resolve(),
    ]
    mock_discover.return_value = mock_files

    # Mock reference data
    mock_load_ref.return_value = {
        Path("/music/track1.mp3").resolve(): 128.0,
        Path("/music/track2.flac").resolve(): 140.0,
    }

    # Mock BPM analysis - one success, one failure
    mock_analyze.side_effect = [
        {"bpm": 127.8, "confidence": 0.95},
        Exception("Analysis failed"),
    ]

    # Run evaluation
    output_dir = tmp_path / "results"
    results = evaluate_bpm(
        source_path=Path("/music"),
        reference_source="metadata",
        sample_size=100,
        output_dir=output_dir,
        full=False,
    )

    # Check summary
    assert results["summary"]["successful"] == 1
    assert results["summary"]["failed"] == 1

    # Check results
    assert results["results"][0]["success"] is True
    assert results["results"][1]["success"] is False
    assert results["results"][1]["error_message"] is not None


@patch("edm.evaluation.evaluators.bpm.analyze_bpm")
@patch("edm.evaluation.evaluators.bpm.load_reference_auto")
@patch("edm.evaluation.evaluators.bpm.discover_audio_files")
def test_evaluate_bpm_full_dataset(mock_discover, mock_load_ref, mock_analyze, tmp_path):
    """Test BPM evaluation with full dataset (no sampling)."""
    # Mock file discovery - 10 files
    mock_files = [Path(f"/music/track{i}.mp3").resolve() for i in range(10)]
    mock_discover.return_value = mock_files

    # Mock reference data
    mock_load_ref.return_value = {f: 128.0 for f in mock_files}

    # Mock BPM analysis
    mock_analyze.return_value = {"bpm": 127.8, "confidence": 0.95}

    # Run evaluation with --full flag
    output_dir = tmp_path / "results"
    results = evaluate_bpm(
        source_path=Path("/music"),
        reference_source="metadata",
        sample_size=5,  # Should be ignored
        output_dir=output_dir,
        full=True,  # Use all files
    )

    # Should evaluate all 10 files, not just 5
    assert results["metadata"]["sample_size"] == 10
    assert results["metadata"]["sampling_strategy"] == "full"
    assert len(results["results"]) == 10


def test_evaluate_bpm_no_files(tmp_path):
    """Test evaluation with no audio files."""
    with patch("edm.evaluation.evaluators.bpm.discover_audio_files") as mock_discover:
        mock_discover.return_value = []

        with pytest.raises(ValueError, match="No audio files found"):
            evaluate_bpm(
                source_path=tmp_path,
                reference_source="metadata",
            )


@patch("edm.evaluation.evaluators.bpm.load_reference_auto")
@patch("edm.evaluation.evaluators.bpm.discover_audio_files")
def test_evaluate_bpm_no_reference(mock_discover, mock_load_ref, tmp_path):
    """Test evaluation with no reference data."""
    # Mock file discovery
    mock_discover.return_value = [Path("/music/track1.mp3")]

    # Mock empty reference data
    mock_load_ref.return_value = {}

    with pytest.raises(ValueError, match="No reference data loaded"):
        evaluate_bpm(
            source_path=tmp_path,
            reference_source="metadata",
        )


@patch("edm.evaluation.evaluators.bpm.analyze_bpm")
@patch("edm.evaluation.evaluators.bpm.load_reference_auto")
@patch("edm.evaluation.evaluators.bpm.discover_audio_files")
def test_evaluate_bpm_output_files_created(mock_discover, mock_load_ref, mock_analyze, tmp_path):
    """Test that output files are created."""
    # Mock file discovery
    mock_files = [Path("/music/track1.mp3").resolve()]
    mock_discover.return_value = mock_files

    # Mock reference data
    mock_load_ref.return_value = {Path("/music/track1.mp3").resolve(): 128.0}

    # Mock BPM analysis
    mock_analyze.return_value = {"bpm": 127.8, "confidence": 0.95}

    # Run evaluation
    output_dir = tmp_path / "results"
    evaluate_bpm(
        source_path=Path("/music"),
        reference_source="metadata",
        output_dir=output_dir,
    )

    # Check that output directory exists
    assert output_dir.exists()

    # Check that at least one JSON file was created
    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) >= 1

    # Check that symlinks were created
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"
    assert latest_json.exists() or latest_json.is_symlink()
    assert latest_md.exists() or latest_md.is_symlink()
