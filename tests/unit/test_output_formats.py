"""Unit tests for output formatting (JSON, YAML)."""

import json
from io import StringIO

import yaml
from rich.console import Console

from cli.commands.analyze import TrackAnalysis, output_json, output_yaml


class TestTrackAnalysis:
    """Tests for TrackAnalysis dataclass."""

    def test_to_dict_basic(self):
        """Test basic conversion to dict."""
        analysis = TrackAnalysis(
            file="track.flac",
            duration=342.5,
            bpm=130.4,
            time_signature="4/4",
            downbeat=0.231,
            structure=[[1, 4, "intro"], [5, 36, "drop"]],
        )
        result = analysis.to_dict()

        assert result["file"] == "track.flac"
        assert result["duration"] == 342.5
        assert result["bpm"] == 130.4
        assert result["time_signature"] == "4/4"
        assert result["structure"] == [[1, 4, "intro"], [5, 36, "drop"]]

    def test_to_dict_with_error(self):
        """Test conversion with error."""
        analysis = TrackAnalysis(
            file="track.flac",
            error="Failed to load audio",
            time=0.5,
        )
        result = analysis.to_dict()

        assert result["file"] == "track.flac"
        assert result["error"] == "Failed to load audio"
        assert result["time"] == 0.5
        assert "duration" not in result
        assert "bpm" not in result

    def test_to_dict_optional_fields(self):
        """Test that None fields are excluded."""
        analysis = TrackAnalysis(
            file="track.flac",
            bpm=128.0,
        )
        result = analysis.to_dict()

        assert result["file"] == "track.flac"
        assert result["bpm"] == 128.0
        assert "duration" not in result
        assert "structure" not in result
        assert "key" not in result


class TestOutputYaml:
    """Tests for YAML output formatting."""

    def test_single_track_yaml(self, tmp_path):
        """Test YAML output for single track."""
        results = [
            {
                "file": "track.flac",
                "duration": 342.5,
                "bpm": 130.4,
                "time_signature": "4/4",
                "downbeat": 0.231,
                "structure": [[1, 4, "intro"], [5, 36, "drop"]],
            }
        ]

        output_file = tmp_path / "output.yaml"
        console = Console(file=StringIO())

        output_yaml(results, output_file, console, quiet=True)

        content = output_file.read_text()
        parsed = list(yaml.safe_load_all(content))

        assert len(parsed) == 1
        assert parsed[0]["file"] == "track.flac"
        assert parsed[0]["bpm"] == 130.4

    def test_multi_document_yaml(self, tmp_path):
        """Test multi-document YAML output for batch."""
        results = [
            {
                "file": "track1.flac",
                "duration": 342.5,
                "bpm": 130.4,
            },
            {
                "file": "track2.flac",
                "duration": 289.1,
                "bpm": 128.0,
            },
        ]

        output_file = tmp_path / "output.yaml"
        console = Console(file=StringIO())

        output_yaml(results, output_file, console, quiet=True)

        content = output_file.read_text()
        parsed = list(yaml.safe_load_all(content))

        assert len(parsed) == 2
        assert parsed[0]["file"] == "track1.flac"
        assert parsed[1]["file"] == "track2.flac"
        assert "---" in content  # Multi-document separator

    def test_yaml_to_stdout(self):
        """Test YAML output to stdout."""
        results = [{"file": "track.flac", "bpm": 128.0}]

        output = StringIO()
        console = Console(file=output, force_terminal=False)

        output_yaml(results, None, console, quiet=False)

        content = output.getvalue()
        assert "track.flac" in content
        assert "bpm: 128.0" in content


class TestOutputJson:
    """Tests for JSON output formatting."""

    def test_json_new_schema(self, tmp_path):
        """Test JSON output uses flat schema."""
        results = [
            {
                "file": "track.flac",
                "duration": 342.5,
                "bpm": 130.4,
                "time_signature": "4/4",
                "structure": [[1, 4, "intro"], [5, 36, "drop"]],
            }
        ]

        output_file = tmp_path / "output.json"
        console = Console(file=StringIO())

        output_json(results, output_file, console, quiet=True)

        content = output_file.read_text()
        parsed = json.loads(content)

        assert len(parsed) == 1
        assert parsed[0]["file"] == "track.flac"
        assert parsed[0]["bpm"] == 130.4
        assert parsed[0]["structure"] == [[1, 4, "intro"], [5, 36, "drop"]]

    def test_json_to_stdout(self):
        """Test JSON output to stdout."""
        results = [{"file": "track.flac", "bpm": 128.0}]

        output = StringIO()
        console = Console(file=output, force_terminal=False)

        output_json(results, None, console, quiet=False)

        content = output.getvalue()
        parsed = json.loads(content)
        assert parsed[0]["bpm"] == 128.0
