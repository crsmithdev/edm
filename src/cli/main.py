"""Main CLI entry point using Typer."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from cli.commands.analyze import analyze_command
from edm import __version__ as lib_version
from edm.config import get_default_log_dir

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="edm",
    help="EDM track analysis CLI",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"EDM CLI v{lib_version}")
        console.print(f"Library v{lib_version}")
        raise typer.Exit()


def setup_logging(verbose: bool = False, quiet: bool = False, no_color: bool = False):
    """Configure logging system.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level logging.
    quiet : bool
        Suppress all logging except errors.
    no_color : bool
        Disable colored output in logs.
    """
    log_dir = get_default_log_dir()
    log_file = log_dir / "edm.log"

    # Determine log level
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            RichHandler(console=Console(stderr=True, no_color=no_color), show_path=False),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging to {log_file}")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
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
    files: List[Path] = typer.Argument(
        ...,
        help="Audio files to analyze",
        exists=True,
    ),
    types: Optional[str] = typer.Option(
        None,
        "--types",
        "-t",
        help="Comma-separated list of analysis types (bpm,grid,structure)",
    ),
    output: Optional[Path] = typer.Option(
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
    config: Optional[Path] = typer.Option(
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging (DEBUG level)",
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
    # Set up logging
    setup_logging(verbose=verbose, quiet=quiet, no_color=no_color)

    # Disable colors if requested or not a TTY
    if no_color or not sys.stdout.isatty():
        console.no_color = True

    # Parse analysis types
    analysis_types = None
    if types:
        analysis_types = [t.strip() for t in types.split(",")]

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
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Analysis failed: {e}", exc_info=verbose)
        if not quiet:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
