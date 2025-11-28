"""Main CLI entry point using Typer."""

import sys
from pathlib import Path

import structlog
import typer
from rich.console import Console

from cli.commands.analyze import analyze_command
from cli.commands.evaluate import evaluate_app
from edm import __version__ as lib_version
from edm.io.audio import clear_audio_cache, set_cache_size
from edm.logging import configure_logging
from edm.processing.parallel import get_default_workers

app = typer.Typer(
    name="edm",
    help="EDM track analysis CLI",
    add_completion=False,
)

# Add subcommands
app.add_typer(evaluate_app, name="evaluate")

console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"EDM CLI v{lib_version}")
        console.print(f"Library v{lib_version}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """EDM track analysis CLI."""
    pass


@app.command()
def analyze(
    files: list[Path] = typer.Argument(
        ...,
        help="Audio files to analyze",
        exists=True,
    ),
    types: str | None = typer.Option(
        None,
        "--types",
        "-t",
        help="Comma-separated list of analysis types (bpm,grid,structure)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSON file",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively analyze directories",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Skip network lookups (Spotify API, etc.)",
    ),
    ignore_metadata: bool = typer.Option(
        False,
        "--ignore-metadata",
        help="Skip reading metadata from audio files",
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        help="Set log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Write logs to file (JSON format)",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Output logs in JSON format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging (equivalent to --log-level DEBUG)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable colored output",
    ),
    cache_size: int = typer.Option(
        10,
        "--cache-size",
        help="Number of audio files to cache in memory (0 to disable)",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers for analysis (default: CPU count - 1)",
    ),
    structure_detector: str = typer.Option(
        "auto",
        "--structure-detector",
        help="Structure detection method: auto (default), msaf, or energy",
    ),
):
    """Analyze EDM tracks for BPM, structure, and other features.

    BPM Lookup Strategy:

    By default, EDM uses a cascading strategy for BPM detection:
    1. File metadata (ID3, MP4, FLAC tags) - fastest
    2. Spotify API - professional accuracy
    3. Computed analysis - most accurate fallback

    Use flags to control the strategy:

    --offline: Skip Spotify API lookups (metadata → computed)

    --ignore-metadata: Skip file metadata (Spotify → computed)

    --offline --ignore-metadata: Force computation only

    Examples:

        edm analyze track.mp3

        edm analyze track.mp3 --types bpm,grid

        edm analyze *.mp3 --output results.json

        edm analyze /path/to/tracks/ --recursive

        edm analyze track.mp3 --offline  # Skip Spotify API

        edm analyze track.mp3 --ignore-metadata  # Force Spotify or compute

        edm analyze track.mp3 --offline --ignore-metadata  # Compute only
    """
    # Determine log level (--verbose overrides --log-level)
    effective_log_level = "DEBUG" if verbose else log_level.upper()
    if quiet:
        effective_log_level = "ERROR"

    # Set up logging
    configure_logging(
        level=effective_log_level,
        json_format=json_logs,
        log_file=log_file,
        no_color=no_color,
    )

    # Configure worker process logging
    from edm.processing.parallel import set_worker_log_level

    set_worker_log_level(effective_log_level)

    logger = structlog.get_logger(__name__)
    logger.debug("cli started", log_level=effective_log_level, json_logs=json_logs)

    # Set default workers if not specified
    if workers is None:
        workers = get_default_workers()
        logger.debug("using default workers", workers=workers)

    # Disable colors if requested or not a TTY
    if no_color or not sys.stdout.isatty():
        console.no_color = True

    # Parse analysis types
    analysis_types = None
    if types:
        analysis_types = [t.strip() for t in types.split(",")]

    # Configure audio cache
    set_cache_size(cache_size)
    logger.debug("audio cache configured", cache_size=cache_size)

    # Run analysis command
    try:
        analyze_command(
            files=files,
            analysis_types=analysis_types,
            output=output,
            format=format,
            config_path=config,
            recursive=recursive,
            offline=offline,
            ignore_metadata=ignore_metadata,
            quiet=quiet,
            console=console,
            workers=workers,
            structure_detector=structure_detector,
        )
    except Exception as e:
        logger = structlog.get_logger(__name__)
        logger.error("analysis failed", error=str(e), exc_info=verbose)
        if not quiet:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    finally:
        # Clear cache to free memory
        clear_audio_cache()


if __name__ == "__main__":
    app()
