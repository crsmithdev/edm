"""Analyze command implementation."""

import json
import time
from pathlib import Path

import structlog
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure import analyze_structure
from edm.config import load_config
from edm.exceptions import AnalysisError, AudioFileError
from edm.processing.parallel import ParallelProcessor

logger = structlog.get_logger(__name__)


# Worker function for parallel execution (must be top-level for pickling)
def _analyze_file_worker(args: tuple) -> dict:
    """Worker function for parallel file analysis.

    Args:
        args: Tuple of (filepath, run_bpm, run_structure, offline, ignore_metadata).

    Returns:
        Analysis result dict with file path and timing.
    """
    filepath, run_bpm, run_structure, offline, ignore_metadata = args

    # Convert string back to Path if needed (for pickling)
    if isinstance(filepath, str):
        filepath = Path(filepath)

    start_time = time.time()

    try:
        result = _analyze_file_impl(filepath, run_bpm, run_structure, offline, ignore_metadata)
        elapsed = time.time() - start_time
        result["file"] = str(filepath)
        result["time"] = elapsed
        return result

    except (AudioFileError, AnalysisError) as e:
        return {
            "file": str(filepath),
            "error": str(e),
            "time": time.time() - start_time,
        }
    except Exception as e:
        return {
            "file": str(filepath),
            "error": f"Unexpected error: {e}",
            "time": time.time() - start_time,
        }


def _analyze_file_impl(
    filepath: Path, run_bpm: bool, run_structure: bool, offline: bool, ignore_metadata: bool
) -> dict:
    """Analyze a single audio file (implementation).

    Args:
        filepath: Path to the audio file.
        run_bpm: Run BPM analysis.
        run_structure: Run structure analysis.
        offline: Skip network lookups.
        ignore_metadata: Skip metadata reading.

    Returns:
        Analysis results.
    """
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


def analyze_command(
    files: list[Path],
    analysis_types: list[str] | None,
    output: Path | None,
    format: str,
    config_path: Path | None,
    recursive: bool,
    offline: bool,
    ignore_metadata: bool,
    quiet: bool,
    console: Console,
    workers: int = 1,
):
    """Execute the analyze command.

    Args:
        files: Audio files or directories to analyze.
        analysis_types: Types of analysis to perform.
        output: Output file path for JSON results.
        format: Output format (table or json).
        config_path: Configuration file path.
        recursive: Recursively process directories.
        offline: Skip network lookups (Spotify API).
        ignore_metadata: Skip reading metadata from files.
        quiet: Suppress non-essential output.
        console: Rich console for output.
        workers: Number of parallel workers (default: 1 for sequential).
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
        workers=workers,
    )
    if not quiet:
        console.print(f"Found {len(audio_files)} file(s) to analyze")
        if workers > 1:
            console.print(f"Using {workers} parallel workers")

    # Prepare args for each file
    args_list = [
        (str(filepath), run_bpm, run_structure, offline, ignore_metadata)
        for filepath in audio_files
    ]

    # Process files
    start_time = time.time()

    # Always use parallel processing (workers=1 is handled by ParallelProcessor)
    results = _process_parallel(args_list, workers, console, quiet)

    total_time = time.time() - start_time

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
            workers=workers,
        )
        console.print(f"\nTotal time: {total_time:.2f}s")
        if len(audio_files) > 1:
            console.print(f"Average time per track: {total_time / len(audio_files):.2f}s")
        if workers > 1:
            # Calculate speedup estimate
            sum_individual = sum(r.get("time", 0) for r in results)
            if sum_individual > 0:
                speedup = sum_individual / total_time
                console.print(f"Parallel speedup: {speedup:.1f}x")


def _process_parallel(
    args_list: list[tuple],
    workers: int,
    console: Console,
    quiet: bool,
) -> list[dict]:
    """Process files in parallel with progress bar.

    Args:
        args_list: List of argument tuples for each file.
        workers: Number of parallel workers.
        console: Rich console for output.
        quiet: Suppress progress output.

    Returns:
        List of analysis results.
    """
    results = []
    completed = [0]  # Use list to allow mutation in callback

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=quiet,
    ) as progress:
        # Use simpler message for single worker
        description = "Analyzing..." if workers == 1 else f"Analyzing ({workers} workers)..."
        task = progress.add_task(description, total=len(args_list))

        def progress_callback(count: int):
            # Update progress bar
            advance = count - completed[0]
            if advance > 0:
                progress.update(task, advance=advance)
                completed[0] = count

        processor = ParallelProcessor(
            worker_fn=_analyze_file_worker,
            workers=workers,
            progress_callback=progress_callback,
        )

        try:
            results = processor.process(args_list)
        except KeyboardInterrupt:
            console.print("\n[yellow]Analysis interrupted[/yellow]")
            raise

    # Report errors after processing
    if not quiet:
        for result in results:
            if "error" in result:
                filepath = Path(result["file"])
                console.print(f"[red]Error analyzing {filepath.name}:[/red] {result['error']}")

    return results


def collect_audio_files(paths: list[Path], recursive: bool) -> list[Path]:
    """Collect all audio files from the given paths.

    Args:
        paths: Files or directories to process.
        recursive: Recursively search directories.

    Returns:
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

    Args:
        filepath: Path to the audio file.
        run_bpm: Run BPM analysis.
        run_structure: Run structure analysis.
        offline: Skip network lookups.
        ignore_metadata: Skip metadata reading.

    Returns:
        Analysis results.
    """
    logger.info("analyzing file", filepath=str(filepath))
    return _analyze_file_impl(filepath, run_bpm, run_structure, offline, ignore_metadata)


def output_json(results: list[dict], output_path: Path | None, console: Console, quiet: bool):
    """Output results as JSON.

    Args:
        results: Analysis results.
        output_path: File to write JSON to (None for stdout).
        console: Rich console for output.
        quiet: Suppress output messages.
    """
    json_str = json.dumps(results, indent=2)

    if output_path:
        output_path.write_text(json_str)
        if not quiet:
            console.print(f"Results written to {output_path}")
    else:
        console.print(json_str)


def output_table(results: list[dict], console: Console, quiet: bool):
    """Output results as a Rich table.

    Args:
        results: Analysis results.
        console: Rich console for output.
        quiet: Suppress output messages.
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
