"""Analyze command implementation."""

import json
import time
from pathlib import Path
from typing import List, Optional

import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure import analyze_structure
from edm.config import load_config
from edm.exceptions import AnalysisError, AudioFileError

logger = structlog.get_logger(__name__)


def analyze_command(
    files: List[Path],
    analysis_types: Optional[List[str]],
    output: Optional[Path],
    format: str,
    config_path: Optional[Path],
    recursive: bool,
    offline: bool,
    ignore_metadata: bool,
    quiet: bool,
    console: Console,
):
    """Execute the analyze command.

    Parameters
    ----------
    files : List[Path]
        Audio files or directories to analyze.
    analysis_types : Optional[List[str]]
        Types of analysis to perform.
    output : Optional[Path]
        Output file path for JSON results.
    format : str
        Output format (table or json).
    config_path : Optional[Path]
        Configuration file path.
    recursive : bool
        Recursively process directories.
    offline : bool
        Skip network lookups (Spotify API).
    ignore_metadata : bool
        Skip reading metadata from files.
    quiet : bool
        Suppress non-essential output.
    console : Console
        Rich console for output.
    """
    # Load configuration
    load_config(config_path)

    # Determine which analyses to run
    run_bpm = analysis_types is None or "bpm" in analysis_types
    run_structure = (
        analysis_types is None or "structure" in analysis_types or "grid" in analysis_types
    )

    # Collect all audio files
    audio_files = collect_audio_files(files, recursive)

    if not audio_files:
        logger.warning("no audio files found", paths=[str(f) for f in files])
        console.print("[yellow]No audio files found[/yellow]")
        return

    logger.info(
        "starting analysis",
        file_count=len(audio_files),
        run_bpm=run_bpm,
        run_structure=run_structure,
        offline=offline,
    )
    if not quiet:
        console.print(f"Found {len(audio_files)} file(s) to analyze")

    # Analyze each file
    results = []
    total_time = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(audio_files))

        for filepath in audio_files:
            try:
                start_time = time.time()
                result = analyze_file(filepath, run_bpm, run_structure, offline, ignore_metadata)
                elapsed = time.time() - start_time
                total_time += elapsed

                result["file"] = str(filepath)
                result["time"] = elapsed
                results.append(result)

                progress.update(task, advance=1)

            except (AudioFileError, AnalysisError) as e:
                logger.error("file analysis failed", filepath=str(filepath), error=str(e))
                if not quiet:
                    console.print(f"[red]Error analyzing {filepath.name}:[/red] {e}")
                results.append(
                    {
                        "file": str(filepath),
                        "error": str(e),
                    }
                )

    # Display results
    if format == "json" or output:
        output_json(results, output, console, quiet)
    else:
        output_table(results, console, quiet)

    # Display timing summary
    if not quiet and len(audio_files) > 0:
        logger.info(
            "analysis complete",
            total_files=len(audio_files),
            total_time=round(total_time, 2),
            avg_time=round(total_time / len(audio_files), 2) if audio_files else 0,
        )
        console.print(f"\nTotal time: {total_time:.2f}s")
        if len(audio_files) > 1:
            console.print(f"Average time per track: {total_time / len(audio_files):.2f}s")


def collect_audio_files(paths: List[Path], recursive: bool) -> List[Path]:
    """Collect all audio files from the given paths.

    Parameters
    ----------
    paths : List[Path]
        Files or directories to process.
    recursive : bool
        Recursively search directories.

    Returns
    -------
    List[Path]
        List of audio file paths.
    """
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []

    for path in paths:
        if path.is_file():
            if path.suffix.lower() in audio_extensions:
                audio_files.append(path)
        elif path.is_dir():
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"

            for ext in audio_extensions:
                audio_files.extend(path.glob(f"{pattern}{ext}"))

    return sorted(audio_files)


def analyze_file(
    filepath: Path, run_bpm: bool, run_structure: bool, offline: bool, ignore_metadata: bool
) -> dict:
    """Analyze a single audio file.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    run_bpm : bool
        Run BPM analysis.
    run_structure : bool
        Run structure analysis.
    offline : bool
        Skip network lookups.
    ignore_metadata : bool
        Skip metadata reading.

    Returns
    -------
    dict
        Analysis results.
    """
    logger.info("analyzing file", filepath=str(filepath))
    result = {}

    if run_bpm:
        bpm_result = analyze_bpm(filepath, offline=offline, ignore_metadata=ignore_metadata)
        result["bpm"] = round(bpm_result.bpm, 1)
        result["bpm_confidence"] = round(bpm_result.confidence, 2)
        result["bpm_source"] = bpm_result.source
        result["bpm_method"] = bpm_result.method
        result["bpm_computation_time"] = round(bpm_result.computation_time, 3)
        if bpm_result.alternatives:
            result["bpm_alternatives"] = [round(alt, 1) for alt in bpm_result.alternatives]

    if run_structure:
        structure_result = analyze_structure(filepath)
        result["duration"] = round(structure_result.duration, 1)
        result["sections"] = len(structure_result.sections)
        result["structure"] = [
            {
                "label": s.label,
                "start": round(s.start_time, 1),
                "end": round(s.end_time, 1),
            }
            for s in structure_result.sections
        ]

    return result


def output_json(results: List[dict], output_path: Optional[Path], console: Console, quiet: bool):
    """Output results as JSON.

    Parameters
    ----------
    results : List[dict]
        Analysis results.
    output_path : Optional[Path]
        File to write JSON to (None for stdout).
    console : Console
        Rich console for output.
    quiet : bool
        Suppress output messages.
    """
    json_str = json.dumps(results, indent=2)

    if output_path:
        output_path.write_text(json_str)
        if not quiet:
            console.print(f"Results written to {output_path}")
    else:
        console.print(json_str)


def output_table(results: List[dict], console: Console, quiet: bool):
    """Output results as a Rich table.

    Parameters
    ----------
    results : List[dict]
        Analysis results.
    console : Console
        Rich console for output.
    quiet : bool
        Suppress output messages.
    """
    if quiet:
        return

    table = Table(title="Analysis Results")
    table.add_column("File", style="cyan")
    table.add_column("BPM", justify="right")
    table.add_column("Source", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Sections", justify="right")
    table.add_column("Time", justify="right")

    # Map sources to icons and colors
    source_icons = {"metadata": "📄", "spotify": "🎵", "computed": "🔬"}
    source_colors = {"metadata": "blue", "spotify": "green", "computed": "yellow"}

    for result in results:
        if "error" in result:
            table.add_row(
                Path(result["file"]).name,
                "[red]Error[/red]",
                "",
                "",
                "",
                "",
                "",
            )
        else:
            # Format BPM source with icon and color
            bpm_source = result.get("bpm_source", "")
            source_icon = source_icons.get(bpm_source, "")
            source_color = source_colors.get(bpm_source, "white")
            source_display = (
                f"{source_icon} [{source_color}]{bpm_source}[/{source_color}]" if bpm_source else ""
            )

            table.add_row(
                Path(result["file"]).name,
                f"{result.get('bpm', 'N/A')}",
                source_display,
                f"{result.get('bpm_confidence', 'N/A')}",
                f"{result.get('duration', 'N/A')}s",
                str(result.get("sections", "N/A")),
                f"{result.get('time', 0):.2f}s",
            )

    console.print(table)
