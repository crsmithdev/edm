"""Audio file serving endpoints."""

from flask import Blueprint, current_app, jsonify, send_file

bp = Blueprint("audio", __name__, url_prefix="/api")


@bp.route("/audio/<path:filename>")
def serve_audio(filename: str):
    """Serve audio file for playback.

    Args:
        filename: Audio filename

    Returns:
        Audio file binary data or 404 error
    """
    audio_service = current_app.audio_service

    try:
        audio_path = audio_service.validate_audio_path(filename)
        return send_file(audio_path)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
