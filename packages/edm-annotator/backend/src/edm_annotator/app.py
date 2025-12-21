"""EDM Structure Annotator - Application factory.

Lightweight Flask app for annotating track structure boundaries.
Outputs YAML in the format expected by the EDM training pipeline.
"""

import argparse
import time
from pathlib import Path

from flask import Flask, g, render_template, request, send_from_directory
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

    # Register request logging middleware (development only)
    if app.debug:
        register_request_logging(app)

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


def register_request_logging(app: Flask) -> None:
    """Register request logging middleware for development.

    Args:
        app: Flask application instance
    """
    # ANSI color codes
    green = "\033[92m"
    cyan = "\033[96m"
    yellow = "\033[93m"
    red = "\033[91m"
    reset = "\033[0m"
    dim = "\033[2m"

    # Method colors
    method_colors = {
        "GET": cyan,
        "POST": green,
        "PUT": yellow,
        "PATCH": yellow,
        "DELETE": red,
        "OPTIONS": dim,
    }

    @app.before_request
    def log_request_start():
        """Log request start and track timing."""
        g.start_time = time.time()

    @app.after_request
    def log_request_end(response):
        """Log request completion with timing and status."""
        # Skip logging for static files and assets
        if request.path.startswith("/assets/") or request.path.startswith("/static/"):
            return response

        duration_ms = (time.time() - g.start_time) * 1000
        method = request.method
        path = request.path
        status = response.status_code

        # Color based on status code
        if status < 300:
            status_color = green
        elif status < 400:
            status_color = cyan
        elif status < 500:
            status_color = yellow
        else:
            status_color = red

        # Color based on method
        method_color = method_colors.get(method, reset)

        # Format duration with color based on speed
        if duration_ms < 50:
            duration_color = green
        elif duration_ms < 200:
            duration_color = yellow
        else:
            duration_color = red

        # Log format: METHOD /path [STATUS] duration_ms
        print(
            f"  {dim}[api]{reset}      "
            f"{method_color}{method:<7}{reset} "
            f"{cyan}{path:<40}{reset} "
            f"[{status_color}{status}{reset}] "
            f"{duration_color}{duration_ms:>6.1f}ms{reset}"
        )

        return response


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

    # Print rich startup information
    _print_startup_banner(app, args)

    app.run(host=args.host, port=args.port, debug=app.debug)


def _print_startup_banner(app: Flask, args) -> None:
    """Print colorized startup banner with configuration details."""
    # ANSI color codes
    bold = "\033[1m"
    green = "\033[92m"
    blue = "\033[94m"
    cyan = "\033[96m"
    yellow = "\033[93m"
    reset = "\033[0m"
    dim = "\033[2m"

    # Count audio files
    audio_dir = Path(app.config["AUDIO_DIR"])
    audio_count = 0
    if audio_dir.exists():
        for ext in app.config["AUDIO_EXTENSIONS"]:
            audio_count += len(list(audio_dir.glob(ext)))

    # Count existing annotations
    annotation_dir = Path(app.config["ANNOTATION_DIR"])
    annotation_count = 0
    if annotation_dir.exists():
        annotation_count = len(list(annotation_dir.glob("**/*.yaml")))

    banner_line = f"{bold}{cyan}╔══════════════════════════════════════════════════════════╗{reset}"
    print(f"\n{banner_line}")
    title = f"{bold}🎵  EDM Structure Annotator - Backend API{reset}"
    print(f"{bold}{cyan}║{reset}  {title}          {bold}{cyan}║{reset}")
    print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════╝{reset}\n")

    # Configuration section
    print(f"{bold}{blue}Configuration{reset}")
    box_top = f"  {dim}┌────────────────────────────────────────────────────────┐{reset}"
    print(box_top)
    print(f"  {dim}│{reset} Environment     {green}{args.env:<39}{reset} {dim}│{reset}")
    debug_status = "enabled" if app.debug else "disabled"
    debug_color = yellow if app.debug else green
    print(f"  {dim}│{reset} Debug Mode      {debug_color}{debug_status:<39}{reset} {dim}│{reset}")
    reload_status = "active" if app.debug else "inactive"
    reload_color = green if app.debug else dim
    print(f"  {dim}│{reset} Auto-reload     {reload_color}{reload_status:<39}{reset} {dim}│{reset}")
    box_bottom = f"  {dim}└────────────────────────────────────────────────────────┘{reset}"
    print(f"{box_bottom}\n")

    # Paths section
    print(f"{bold}{blue}Paths{reset}")
    print(box_top)
    audio_path_str = str(audio_dir)[:52]
    print(f"  {dim}│{reset} Audio           {cyan}{audio_path_str:<39}{reset} {dim}│{reset}")
    annotation_path_str = str(annotation_dir)[:52]
    print(f"  {dim}│{reset} Annotations     {cyan}{annotation_path_str:<39}{reset} {dim}│{reset}")
    print(f"{box_bottom}\n")

    # Data statistics section
    print(f"{bold}{blue}Available Data{reset}")
    print(box_top)
    audio_color = green if audio_count > 0 else yellow
    audio_files_line = f"  {dim}│{reset} Audio Files     {audio_color}{audio_count:<39}{reset}"
    print(f"{audio_files_line} {dim}│{reset}")
    annotation_color = green if annotation_count > 0 else dim
    anno_line = f"  {dim}│{reset} Annotations     {annotation_color}{annotation_count:<39}{reset}"
    print(f"{anno_line} {dim}│{reset}")
    formats = [ext.replace("*", "") for ext in app.config["AUDIO_EXTENSIONS"]]
    supported_formats = ", ".join(formats)
    formats_line = f"  {dim}│{reset} Formats         {dim}{supported_formats:<39}{reset}"
    print(f"{formats_line} {dim}│{reset}")
    print(f"{box_bottom}\n")

    # Server info
    print(f"{bold}{blue}Server{reset}")
    print(box_top)
    server_url = f"http://{args.host}:{args.port}"
    print(f"  {dim}│{reset} URL             {bold}{green}{server_url:<39}{reset} {dim}│{reset}")
    cors_origins = ", ".join(app.config.get("CORS_ORIGINS", []))
    cors_display = cors_origins if cors_origins else "disabled"
    print(f"  {dim}│{reset} CORS            {dim}{cors_display:<39}{reset} {dim}│{reset}")
    print(f"{box_bottom}\n")

    # Status and next steps
    if audio_count == 0:
        warning = f"  {yellow}⚠{reset}  No audio files found. "
        instruction = f"Set {cyan}EDM_AUDIO_DIR{reset} or add files to {cyan}{audio_dir}{reset}"
        print(warning + instruction)
    else:
        print(f"  {green}✓{reset}  Ready to annotate {green}{audio_count}{reset} tracks")

    print(f"\n  {dim}Press Ctrl+C to stop the server{reset}\n")


if __name__ == "__main__":
    main()
