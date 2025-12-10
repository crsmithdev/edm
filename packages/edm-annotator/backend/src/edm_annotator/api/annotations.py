"""Annotation save/load endpoints."""

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("annotations", __name__, url_prefix="/api")


@bp.route("/save", methods=["POST"])
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

    Returns:
        JSON object with:
            - success: True if saved
            - output: Path to saved file
            - boundaries_count: Number of boundaries saved
    """
    data = request.json

    # Validate request data
    if not data or "filename" not in data or "boundaries" not in data:
        return jsonify({"error": "Invalid data"}), 400

    filename = data["filename"]
    bpm = data.get("bpm", 128.0)
    downbeat = data.get("downbeat", 0.0)
    boundaries = data["boundaries"]

    annotation_service = current_app.annotation_service

    try:
        output_file = annotation_service.save_annotation(filename, bpm, downbeat, boundaries)

        return jsonify(
            {
                "success": True,
                "output": str(output_file),
                "boundaries_count": len(boundaries),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error saving annotation: {str(e)}"}), 500
