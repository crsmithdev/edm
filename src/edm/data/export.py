"""Export utilities for ML frameworks."""

import json
from pathlib import Path

from edm.data.schema import Annotation


def export_to_json(annotations: list[Annotation], output_path: Path) -> None:
    """Export annotations to JSON format.

    Args:
        annotations: List of annotations to export
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [ann.model_dump(mode="json", exclude_none=True) for ann in annotations]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_to_pytorch(annotations: list[Annotation], output_path: Path) -> None:
    """Export annotations to PyTorch-compatible format.

    Creates a dictionary with keys suitable for PyTorch Dataset loading:
    - file_paths: List of audio file paths
    - bpms: List of BPM values
    - structures: List of structure annotations
    - metadata: List of metadata dictionaries

    Args:
        annotations: List of annotations to export
        output_path: Path to save PyTorch data file (.pt or .json)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "file_paths": [str(ann.audio.file) for ann in annotations],
        "bpms": [ann.audio.bpm for ann in annotations],
        "durations": [ann.audio.duration for ann in annotations],
        "downbeats": [ann.audio.downbeat for ann in annotations],
        "structures": [
            [
                {"bar": sec.bar, "label": sec.label, "time": sec.time, "confidence": sec.confidence}
                for sec in ann.structure
            ]
            for ann in annotations
        ],
        "metadata": [
            {
                "tier": ann.metadata.tier,
                "confidence": ann.metadata.confidence,
                "source": ann.metadata.source,
            }
            for ann in annotations
        ],
    }

    # Save as JSON (can be loaded by PyTorch Dataset)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_to_tensorflow(annotations: list[Annotation], output_path: Path) -> None:
    """Export annotations to TensorFlow-compatible format.

    Creates a JSON file that can be loaded into tf.data.Dataset.

    Args:
        annotations: List of annotations to export
        output_path: Path to save TensorFlow data file (.json)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TensorFlow prefers flat structure for tf.data.Dataset.from_tensor_slices
    data = {
        "examples": [
            {
                "audio_file": str(ann.audio.file),
                "bpm": ann.audio.bpm,
                "duration": ann.audio.duration,
                "downbeat": ann.audio.downbeat,
                "structure": [
                    {
                        "bar": sec.bar,
                        "label": sec.label,
                        "time": sec.time,
                        "confidence": sec.confidence,
                    }
                    for sec in ann.structure
                ],
                "tier": ann.metadata.tier,
                "confidence": ann.metadata.confidence,
                "source": ann.metadata.source,
            }
            for ann in annotations
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_to_csv(annotations: list[Annotation], output_path: Path, format: str = "time") -> None:
    """Export annotations to CSV format for evaluation.

    Args:
        annotations: List of annotations to export
        output_path: Path to save CSV file
        format: Export format - "time" (time-based) or "bar" (bar-based)
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        if format == "time":
            fieldnames = ["filename", "start", "end", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ann in annotations:
                filename = ann.audio.file.name
                for i in range(len(ann.structure) - 1):
                    writer.writerow(
                        {
                            "filename": filename,
                            "start": ann.structure[i].time,
                            "end": ann.structure[i + 1].time,
                            "label": ann.structure[i].label,
                        }
                    )
                # Last section goes to end of track
                if ann.structure:
                    writer.writerow(
                        {
                            "filename": filename,
                            "start": ann.structure[-1].time,
                            "end": ann.audio.duration,
                            "label": ann.structure[-1].label,
                        }
                    )

        elif format == "bar":
            fieldnames = ["filename", "start_bar", "end_bar", "label", "bpm"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ann in annotations:
                filename = ann.audio.file.name
                # Calculate end bar from duration and BPM
                total_beats = ann.audio.duration * (ann.audio.bpm / 60)
                end_bar = int(total_beats / 4) + 1  # 4 beats per bar in 4/4 time

                for i in range(len(ann.structure) - 1):
                    writer.writerow(
                        {
                            "filename": filename,
                            "start_bar": ann.structure[i].bar,
                            "end_bar": ann.structure[i + 1].bar,
                            "label": ann.structure[i].label,
                            "bpm": ann.audio.bpm,
                        }
                    )
                # Last section goes to end of track
                if ann.structure:
                    writer.writerow(
                        {
                            "filename": filename,
                            "start_bar": ann.structure[-1].bar,
                            "end_bar": end_bar,
                            "label": ann.structure[-1].label,
                            "bpm": ann.audio.bpm,
                        }
                    )

        else:
            raise ValueError(f"Unknown format: {format}. Use 'time' or 'bar'")
