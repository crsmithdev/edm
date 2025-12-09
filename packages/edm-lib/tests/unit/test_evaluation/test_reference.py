"""Tests for evaluation.reference module."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from edm.evaluation.reference import (
    load_metadata_reference,
    load_reference_auto,
    load_reference_csv,
    load_reference_json,
)


def test_load_reference_csv(tmp_path):
    """Test loading reference from CSV file."""
    csv_path = tmp_path / "reference.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bpm"])
        writer.writerow(["/music/track1.mp3", "128.0"])
        writer.writerow(["/music/track2.flac", "140.0"])

    reference = load_reference_csv(csv_path, value_field="bpm")

    assert len(reference) == 2
    assert reference[Path("/music/track1.mp3").resolve()] == 128.0
    assert reference[Path("/music/track2.flac").resolve()] == 140.0


def test_load_reference_csv_missing_column(tmp_path):
    """Test loading CSV with missing required column."""
    csv_path = tmp_path / "reference.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tempo"])  # Wrong columns
        writer.writerow(["/music/track1.mp3", "128.0"])

    with pytest.raises(ValueError, match="must have 'path' column"):
        load_reference_csv(csv_path, value_field="bpm")


def test_load_reference_json(tmp_path):
    """Test loading reference from JSON file."""
    json_path = tmp_path / "reference.json"

    data = [
        {"path": "/music/track1.mp3", "bpm": 128.0},
        {"path": "/music/track2.flac", "bpm": 140.0},
    ]

    with open(json_path, "w") as f:
        json.dump(data, f)

    reference = load_reference_json(json_path, value_field="bpm")

    assert len(reference) == 2
    assert reference[Path("/music/track1.mp3").resolve()] == 128.0
    assert reference[Path("/music/track2.flac").resolve()] == 140.0


def test_load_reference_json_invalid_format(tmp_path):
    """Test loading JSON with invalid format."""
    json_path = tmp_path / "reference.json"

    # Not a list
    with open(json_path, "w") as f:
        json.dump({"path": "/music/track1.mp3", "bpm": 128.0}, f)

    with pytest.raises(ValueError, match="must be a list"):
        load_reference_json(json_path, value_field="bpm")


@patch("edm.io.metadata.read_metadata")
@patch("edm.evaluation.reference.discover_audio_files")
def test_load_metadata_reference(mock_discover, mock_read_metadata):
    """Test loading reference from file metadata."""
    # Mock file discovery
    mock_files = [Path("/music/track1.mp3"), Path("/music/track2.flac")]
    mock_discover.return_value = mock_files

    # Mock metadata reading
    mock_read_metadata.side_effect = [
        {"bpm": 128.0, "artist": "Artist 1"},
        {"bpm": 140.0, "artist": "Artist 2"},
    ]

    reference = load_metadata_reference(Path("/music"), value_field="bpm")

    assert len(reference) == 2
    assert reference[Path("/music/track1.mp3")] == 128.0
    assert reference[Path("/music/track2.flac")] == 140.0


def test_load_reference_auto_csv(tmp_path):
    """Test auto-loading CSV reference."""
    csv_path = tmp_path / "reference.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bpm"])
        writer.writerow(["/music/track1.mp3", "128.0"])

    reference = load_reference_auto(
        reference_arg=str(csv_path),
        analysis_type="bpm",
        source_path=Path("/music"),
        value_field="bpm",
    )

    assert len(reference) == 1
    assert reference[Path("/music/track1.mp3").resolve()] == 128.0


def test_load_reference_auto_json(tmp_path):
    """Test auto-loading JSON reference."""
    json_path = tmp_path / "reference.json"

    data = [{"path": "/music/track1.mp3", "bpm": 128.0}]

    with open(json_path, "w") as f:
        json.dump(data, f)

    reference = load_reference_auto(
        reference_arg=str(json_path),
        analysis_type="bpm",
        source_path=Path("/music"),
        value_field="bpm",
    )

    assert len(reference) == 1


def test_load_reference_auto_metadata_unsupported():
    """Test that metadata reference is rejected for unsupported analysis types."""
    with pytest.raises(ValueError, match="Metadata reference not supported"):
        load_reference_auto(
            reference_arg="metadata",
            analysis_type="drops",
            source_path=Path("/music"),
            value_field="drops",
        )


def test_load_reference_auto_invalid_format(tmp_path):
    """Test that invalid reference format raises error."""
    # Create a file with invalid extension
    invalid_file = tmp_path / "file.txt"
    invalid_file.touch()

    with pytest.raises(ValueError, match="Unknown reference format"):
        load_reference_auto(
            reference_arg=str(invalid_file),
            analysis_type="bpm",
            source_path=Path("/music"),
            value_field="bpm",
        )
