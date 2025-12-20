"""Waveform generation endpoints."""

from flask import Blueprint, current_app, jsonify

bp = Blueprint("waveforms", __name__, url_prefix="/api")


@bp.route("/load/<path:filename>")
def load_track(filename: str):
    """Load track waveform data and metadata.

    Generates 3-band waveform (bass/mids/highs) for visualization
    and loads BPM/downbeat/boundaries from existing annotation if available.

    Args:
        filename: Audio filename

    Returns:
        JSON object with:
            - filename: Track filename
            - duration: Duration in seconds
            - bpm: BPM (from annotation if exists, else null)
            - downbeat: Downbeat time in seconds
            - boundaries: List of boundary dicts with 'time' and 'label'
                         (if annotation exists, else null)
            - annotation_tier: 1 for reference (hand-tagged), 2 for generated,
                              null if no annotation
            - sample_rate: Sample rate used
            - waveform_bass: RMS values for bass band (20-250 Hz)
            - waveform_mids: RMS values for mids band (250-4000 Hz)
            - waveform_highs: RMS values for highs band (4000+ Hz)
            - waveform_times: Frame times in seconds
    """
    waveform_service = current_app.waveform_service
    annotation_service = current_app.annotation_service

    try:
        # Generate waveform
        waveform_data = waveform_service.generate_waveform(filename)

        # Load BPM, downbeat, and boundaries from existing annotation
        # Prefers reference (tier 1) over generated (tier 2)
        bpm = None
        downbeat = 0.0
        boundaries = None
        annotation_tier = None
        annotation = annotation_service.load_annotation(filename)
        if annotation:
            bpm = annotation.get("bpm")
            downbeat = annotation.get("downbeat", 0.0)
            boundaries = annotation.get("boundaries")
            annotation_tier = annotation.get("tier")

        return jsonify(
            {
                "filename": filename,
                "duration": waveform_data["duration"],
                "bpm": bpm,
                "downbeat": downbeat,
                "boundaries": boundaries,
                "annotation_tier": annotation_tier,
                "sample_rate": waveform_data["sample_rate"],
                "waveform_bass": waveform_data["waveform_bass"],
                "waveform_mids": waveform_data["waveform_mids"],
                "waveform_highs": waveform_data["waveform_highs"],
                "waveform_times": waveform_data["waveform_times"],
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error loading track: {str(e)}"}), 500
