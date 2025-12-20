"""Main CLI entry point using Typer."""

import sys
from pathlib import Path

import structlog
import typer
from edm import __version__ as lib_version
from edm.io.audio import clear_audio_cache, set_cache_size
from edm.logging import configure_logging
from rich.console import Console

from edm_cli.commands.analyze import analyze_command
from edm_cli.commands.data import app as data_app
from edm_cli.commands.evaluate import evaluate_command
from edm_cli.commands.models import app as models_app
from edm_cli.commands.train import train_command

app = typer.Typer(
    name="edm",
    help="EDM track analysis CLI",
    add_completion=False,
)

# Add subcommands
app.command(name="evaluate")(evaluate_command)
app.command(name="train")(train_command)
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")

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
        help="Comma-separated list of analysis types (bpm,beats,grid,structure)",
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
        help="Output format (table, json, yaml)",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively analyze directories",
    ),
    no_metadata: bool = typer.Option(
        False,
        "--no-metadata",
        help="Skip reading metadata from audio files",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Write logs to file (JSON format)",
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
):
    """Analyze EDM tracks for BPM, structure, and other features.

    BPM Lookup Strategy:

    By default, EDM uses a cascading strategy for BPM detection:
    1. File metadata (ID3, MP4, FLAC tags) - fastest
    2. Computed analysis - most accurate fallback

    Use flags to control the strategy:

    --no-metadata: Force computation only

    Examples:

        edm analyze track.mp3

        edm analyze track.mp3 --types bpm,grid

        edm analyze track.mp3 --types beats  # Beat/downbeat detection

        edm analyze *.mp3 --output results.json

        edm analyze /path/to/tracks/ --recursive

        edm analyze track.mp3 --no-metadata  # Force computation
    """
    # Map verbosity count to log level
    verbosity_map = {
        0: "WARNING",
        1: "INFO",
        2: "DEBUG",
    }
    effective_log_level = verbosity_map.get(verbose, "DEBUG")
    if quiet:
        effective_log_level = "ERROR"

    # Set up logging
    configure_logging(
        level=effective_log_level,
        json_format=False,
        log_file=log_file,
        no_color=no_color,
    )

    # Configure worker process logging
    from edm.processing.parallel import set_worker_log_level

    set_worker_log_level(effective_log_level)

    logger = structlog.get_logger(__name__)
    logger.debug("cli started", log_level=effective_log_level)

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
            config_path=None,  # Config file support not yet implemented
            recursive=recursive,
            offline=False,  # Legacy parameter, no longer used
            ignore_metadata=no_metadata,
            quiet=quiet,
            console=console,
            workers=workers,
            structure_detector="auto",
            annotations=False,  # Legacy parameter, use --format yaml instead
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
