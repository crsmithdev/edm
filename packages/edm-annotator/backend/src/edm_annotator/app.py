"""EDM Structure Annotator - Application factory.

Lightweight Flask app for annotating track structure boundaries.
Outputs YAML in the format expected by the EDM training pipeline.
"""

import argparse
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from .api import register_routes
from .config import config_class_map
from .services import AnnotationService, AudioService, WaveformService


def create_app(config_name: str = "development") -> Flask:
    """Application factory - creates and configures Flask app.

    Args:
        config_name: Configuration to use (development/production/testing)

    Returns:
        Configured Flask application instance
    """
    # Determine template/static directories relative to backend package
    package_root = Path(__file__).parent.parent.parent.parent
    template_dir = package_root / "templates"
    static_dir = package_root / "static"

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

    # Load configuration
    config_class = config_class_map[config_name]
    app.config.from_object(config_class)

    # Enable CORS for development (Vite dev server)
    if app.config.get("CORS_ORIGINS"):
        CORS(app, origins=app.config["CORS_ORIGINS"])

    # Initialize services (dependency injection)
    audio_service = AudioService(config=app.config)
    annotation_service = AnnotationService(config=app.config, audio_service=audio_service)
    waveform_service = WaveformService(config=app.config, audio_service=audio_service)

    # Store services in app context for route access
    app.audio_service = audio_service
    app.annotation_service = annotation_service
    app.waveform_service = waveform_service

    # Register API blueprints
    register_routes(app)

    # Register error handlers
    register_error_handlers(app)

    # Main route - serve frontend
    @app.route("/")
    def index():
        """Main annotation interface."""
        return render_template("index.html")

    # In production, serve built frontend from dist/
    if not app.debug:
        frontend_dist = package_root / "frontend" / "dist"
        if frontend_dist.exists():

            @app.route("/assets/<path:path>")
            def serve_frontend_assets(path):
                """Serve frontend static assets."""
                return send_from_directory(frontend_dist / "assets", path)

    return app


def register_error_handlers(app: Flask) -> None:
    """Register error handlers for common HTTP errors.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500


def main():
    """CLI entry point for 'edm-annotator' command."""
    parser = argparse.ArgumentParser(description="EDM Structure Annotator Web Server")
    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "production", "testing"],
        help="Environment configuration",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    args = parser.parse_args()

    app = create_app(args.env)

    # Print startup information
    print(f"Environment: {args.env}")
    print(f"Audio directory: {app.config['AUDIO_DIR']}")
    print(f"Annotation directory: {app.config['ANNOTATION_DIR']}")
    print(f"\nStarting annotation server on http://{args.host}:{args.port}")
    if app.debug:
        print("Debug mode enabled - auto-reload active")

    app.run(host=args.host, port=args.port, debug=app.debug)


if __name__ == "__main__":
    main()
