"""Simple EDM structure annotation web tool.

Lightweight Flask app for annotating track structure boundaries.
Outputs YAML in the format expected by the EDM training pipeline.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import yaml
from edm.data.metadata import AnnotationMetadata
from edm.data.schema import Annotation, AudioMetadata, StructureSection
from edm.io.audio import load_audio
from flask import Flask, jsonify, render_template, request, send_file

# Configuration - use environment variables for flexibility
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MONOREPO_ROOT = PACKAGE_ROOT.parent.parent
TEMPLATE_DIR = PACKAGE_ROOT / "templates"
STATIC_DIR = PACKAGE_ROOT / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
AUDIO_DIR = Path(os.getenv("EDM_AUDIO_DIR", Path.home() / "music"))
ANNOTATION_DIR = Path(os.getenv("EDM_ANNOTATION_DIR", MONOREPO_ROOT / "data" / "annotations"))
REFERENCE_DIR = ANNOTATION_DIR / "reference"
GENERATED_DIR = ANNOTATION_DIR / "generated"

# Valid EDM section labels
VALID_LABELS = ["intro", "buildup", "breakdown", "breakbuild", "outro", "unlabeled"]


@app.route("/")
def index():
    """Main annotation interface."""
    return render_template("index.html")


@app.route("/api/tracks")
def list_tracks():
    """List available audio files."""
    tracks = []
    for ext in ["*.mp3", "*.flac", "*.wav", "*.m4a"]:
        for path in AUDIO_DIR.glob(ext):
            tracks.append(
                {
                    "filename": path.name,
                    "path": str(path.relative_to(Path.home())),
                    "has_reference": (REFERENCE_DIR / f"{path.stem}.yaml").exists(),
                    "has_generated": (GENERATED_DIR / f"{path.stem}.yaml").exists(),
                }
            )
    return jsonify(sorted(tracks, key=lambda x: x["filename"]))


@app.route("/api/audio/<path:filename>")
def serve_audio(filename):
    """Serve audio file for playback."""
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(audio_path)


@app.route("/api/load/<path:filename>")
def load_track(filename):
    """Load track audio data and metadata.

    Returns waveform data for visualization and BPM estimate.
    """
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Load audio using edm library (with caching)
    y, sr = load_audio(audio_path, sr=22050)
    duration = len(y) / sr

    # Try to load BPM and downbeat from existing annotation
    bpm = None
    downbeat = 0.0
    annotation_path, tier = _find_annotation_for_file(filename)

    if annotation_path and annotation_path.exists():
        try:
            with open(annotation_path, "r") as f:
                annotation_data = yaml.safe_load(f)
                if annotation_data and "audio" in annotation_data:
                    bpm = annotation_data["audio"].get("bpm")
                    downbeat = annotation_data["audio"].get("downbeat", 0.0)
        except Exception as e:
            print(f"Warning: Could not load annotation: {e}")

    # If BPM not in annotation, return None (frontend will prompt user)

    # High-resolution 3-band waveform for beat grid editor
    hop_length = 128  # ~5.8ms at 22050 Hz
    frame_length = 1024

    # Split into 3 frequency bands using bandpass filters
    # Bass: 20-250 Hz, Mids: 250-4000 Hz, Highs: 4000+ Hz
    from scipy import signal

    # Design butterworth bandpass filters
    nyquist = sr / 2

    # Bass band (20-250 Hz)
    bass_low = 20 / nyquist
    bass_high = 250 / nyquist
    b_bass, a_bass = signal.butter(4, [bass_low, bass_high], btype="band")
    y_bass = signal.filtfilt(b_bass, a_bass, y)

    # Mids band (250-4000 Hz)
    mids_low = 250 / nyquist
    mids_high = 4000 / nyquist
    b_mids, a_mids = signal.butter(4, [mids_low, mids_high], btype="band")
    y_mids = signal.filtfilt(b_mids, a_mids, y)

    # Highs band (4000+ Hz)
    highs_low = 4000 / nyquist
    b_highs, a_highs = signal.butter(4, highs_low, btype="high")
    y_highs = signal.filtfilt(b_highs, a_highs, y)

    # Calculate RMS for each band
    rms_bass = librosa.feature.rms(y=y_bass, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mids = librosa.feature.rms(y=y_mids, frame_length=frame_length, hop_length=hop_length)[0]
    rms_highs = librosa.feature.rms(y=y_highs, frame_length=frame_length, hop_length=hop_length)[0]

    # Time axis for RMS frames
    times = librosa.frames_to_time(np.arange(len(rms_bass)), sr=sr, hop_length=hop_length)

    return jsonify(
        {
            "filename": filename,
            "duration": duration,
            "bpm": bpm,
            "downbeat": downbeat,
            "sample_rate": sr,
            "waveform_bass": rms_bass.tolist(),
            "waveform_mids": rms_mids.tolist(),
            "waveform_highs": rms_highs.tolist(),
            "waveform_times": times.tolist(),
        }
    )


@app.route("/api/save", methods=["POST"])
def save_annotation():
    """Save annotation to YAML file.

    Expected JSON format:
    {
        "filename": "track.mp3",
        "bpm": 128.0,
        "downbeat": 0.0,
        "boundaries": [
            {"time": 0.0, "label": "intro"},
            {"time": 15.2, "label": "buildup"},
            ...
        ]
    }
    """
    data = request.json

    if not data or "filename" not in data or "boundaries" not in data:
        return jsonify({"error": "Invalid data"}), 400

    filename = data["filename"]
    bpm = data.get("bpm", 128.0)
    downbeat = data.get("downbeat", 0.0)
    boundaries = data["boundaries"]

    # Build annotation using edm schema
    now = datetime.now(timezone.utc)
    audio_path = AUDIO_DIR / filename

    # Get audio duration
    duration = librosa.get_duration(path=audio_path)

    # Convert timestamps to bars and create StructureSection objects
    structure_sections = []
    for boundary in boundaries:
        time = boundary["time"]
        label = boundary["label"]

        # Calculate bar number (1-indexed)
        # Bar = (time - downbeat) / (60/bpm * 4) + 1
        bar = int((time - downbeat) / (60.0 / bpm * 4.0)) + 1
        bar = max(1, bar)  # Minimum bar 1

        structure_sections.append(StructureSection(bar=bar, label=label, time=time, confidence=1.0))

    # Sort by time
    structure_sections = sorted(structure_sections, key=lambda x: x.time)

    # Build Annotation object using schema
    annotation = Annotation(
        metadata=AnnotationMetadata(
            tier=1,  # Tier 1 = manual annotation
            confidence=1.0,
            source="manual",
            created=now,
            modified=now,
            annotator="web_tool",
            flags=[],
        ),
        audio=AudioMetadata(
            file=audio_path, duration=duration, bpm=bpm, downbeat=downbeat, time_signature=(4, 4)
        ),
        structure=structure_sections,
    )

    # Save using schema's to_yaml method
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = REFERENCE_DIR / f"{Path(filename).stem}.yaml"
    annotation.to_yaml(output_file)

    return jsonify(
        {"success": True, "output": str(output_file), "boundaries_count": len(structure_sections)}
    )


def _find_annotation_for_file(filename: str) -> tuple[Path | None, int | None]:
    """Find annotation YAML file for given audio filename.

    Args:
        filename: Audio filename (e.g., "Artist - Track.flac")

    Returns:
        (annotation_path, tier) or (None, None) if not found
        Prefers reference (tier 1) over generated (tier 2)
    """
    stem = Path(filename).stem

    # Check reference first (tier 1)
    ref_path = REFERENCE_DIR / f"{stem}.yaml"
    if ref_path.exists():
        return (ref_path, 1)

    # Fall back to generated (tier 2)
    gen_path = GENERATED_DIR / f"{stem}.yaml"
    if gen_path.exists():
        return (gen_path, 2)

    # Try case-insensitive search in both directories
    for directory, tier in [(REFERENCE_DIR, 1), (GENERATED_DIR, 2)]:
        if directory.exists():
            for yaml_file in directory.glob("*.yaml"):
                if yaml_file.stem.lower() == stem.lower():
                    return (yaml_file, tier)

    return (None, None)


if __name__ == "__main__":
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Annotation directory: {ANNOTATION_DIR}")
    print("\nStarting annotation server with auto-reload...")
    print("Open http://localhost:5000 in your browser")
    print("Changes to Python files will automatically restart the server")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=True,
        extra_files=["templates/index.html"],
    )
