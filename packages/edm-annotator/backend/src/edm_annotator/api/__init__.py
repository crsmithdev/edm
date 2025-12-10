"""API routes for EDM Annotator."""

from flask import Flask

from . import annotations, audio, tracks, waveforms


def register_routes(app: Flask) -> None:
    """Register all API blueprints.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(tracks.bp)
    app.register_blueprint(audio.bp)
    app.register_blueprint(waveforms.bp)
    app.register_blueprint(annotations.bp)
