"""Track listing endpoints."""

from pathlib import Path

from flask import Blueprint, current_app, jsonify

bp = Blueprint("tracks", __name__, url_prefix="/api")


@bp.route("/tracks")
def list_tracks():
    """List available audio files with annotation status.

    Returns:
        JSON array of track objects with:
            - filename: Track filename
            - path: Relative path from home directory
            - has_reference: Whether reference annotation exists
            - has_generated: Whether generated annotation exists
    """
    audio_service = current_app.audio_service
    annotation_service = current_app.annotation_service

    tracks = []
    for audio_path in audio_service.list_audio_files():
        # Check for existing annotations
        ref_exists = (annotation_service.reference_dir / f"{audio_path.stem}.yaml").exists()
        gen_exists = (annotation_service.generated_dir / f"{audio_path.stem}.yaml").exists()

        # Try to calculate relative path from home, fall back to absolute path
        try:
            path_str = str(audio_path.relative_to(Path.home()))
        except ValueError:
            # Path is not under home directory (e.g., in tests)
            path_str = str(audio_path)

        tracks.append(
            {
                "filename": audio_path.name,
                "path": path_str,
                "has_reference": ref_exists,
                "has_generated": gen_exists,
            }
        )

    return jsonify(sorted(tracks, key=lambda x: x["filename"]))
