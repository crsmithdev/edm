"""Analyze command implementation."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from edm.analysis.bars import bars_to_time
from edm.analysis.beat_detector import detect_beats
from edm.analysis.bpm import analyze_bpm
from edm.analysis.structure import analyze_structure
from edm.config import load_config
from edm.exceptions import AnalysisError, AudioFileError
from edm.io.files import discover_audio_files
from edm.processing.parallel import ParallelProcessor

logger = structlog.get_logger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds as M:SS.ss or MM:SS.ss.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string like '2:32.02s'.
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}s"


@dataclass
class TrackAnalysis:
    """Analysis result for a single track with flat schema."""

    file: str
    duration: float | None = None
    bpm: float | None = None
    downbeat: float | None = None
    time_signature: str | None = None
    key: str | None = None
    structure: list[list] | None = None
    events: list[list] | None = None
    raw: list[dict] | None = None
    error: str | None = None
    time: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary in flat schema format.

        Returns:
            Dictionary with flat structure for YAML/JSON output.
        """
        result: dict = {"file": self.file}

        if self.error:
            result["error"] = self.error
            if self.time is not None:
                result["time"] = self.time
            return result

        if self.duration is not None:
            result["duration"] = self.duration

        if self.bpm is not None:
            result["bpm"] = self.bpm

        if self.downbeat is not None:
            result["downbeat"] = self.downbeat

        if self.time_signature is not None:
            result["time_signature"] = self.time_signature

        if self.key:
            result["key"] = self.key

        if self.structure:
            result["structure"] = self.structure

        if self.events:
            result["events"] = self.events

        if self.raw:
            result["raw"] = self.raw

        return result


# Worker function for parallel execution (must be top-level for pickling)
def _analyze_file_worker(args: tuple) -> dict:
    """Worker function for parallel file analysis.

    Args:
        args: Tuple of (filepath, run_bpm, run_structure, run_beats, offline, ignore_metadata, structure_detector).

    Returns:
        Analysis result dict with file path and timing.
    """
    filepath, run_bpm, run_structure, run_beats, offline, ignore_metadata, structure_detector = args

    # Convert string back to Path if needed (for pickling)
    if isinstance(filepath, str):
        filepath = Path(filepath)

    start_time = time.time()

    try:
        analysis = _analyze_file_impl(
            filepath,
            run_bpm,
            run_structure,
            run_beats,
            offline,
            ignore_metadata,
            structure_detector,
        )
        analysis.time = time.time() - start_time
        return analysis.to_dict()

    except (AudioFileError, AnalysisError) as e:
        return TrackAnalysis(
            file=str(filepath),
            error=str(e),
            time=time.time() - start_time,
        ).to_dict()
    except Exception as e:
        return TrackAnalysis(
            file=str(filepath),
            error=f"Unexpected error: {e}",
            time=time.time() - start_time,
        ).to_dict()


def _analyze_file_impl(
    filepath: Path,
    run_bpm: bool,
    run_structure: bool,
    run_beats: bool,
    offline: bool,
    ignore_metadata: bool,
    structure_detector: str = "auto",
) -> TrackAnalysis:
    """Analyze a single audio file (implementation).

    Args:
        filepath: Path to the audio file.
        run_bpm: Run BPM analysis.
        run_structure: Run structure analysis.
        run_beats: Run beat detection.
        offline: Skip network lookups.
        ignore_metadata: Skip metadata reading.
        structure_detector: Structure detection method (auto, msaf, energy).

    Returns:
        TrackAnalysis with flat schema.
    """
    bpm: float | None = None
    downbeat: float | None = None
    time_signature: str | None = None
    duration: float | None = None
    structure: list[list] | None = None
    events: list[list] | None = None
    raw: list[dict] | None = None

    if run_bpm:
        bpm_result = analyze_bpm(filepath, offline=offline, ignore_metadata=ignore_metadata)
        bpm = round(bpm_result.bpm, 1)

    if run_beats:
        beat_grid = detect_beats(filepath)
        bpm = round(beat_grid.bpm, 1)
        downbeat = round(beat_grid.first_beat_time, 3)
        time_signature = f"{beat_grid.time_signature[0]}/{beat_grid.time_signature[1]}"

    if run_structure:
        structure_result = analyze_structure(filepath, detector=structure_detector)  # type: ignore[arg-type]
        duration = round(structure_result.duration, 1)
        if structure_result.bpm:
            bpm = round(structure_result.bpm, 1)
        if structure_result.downbeat is not None:
            downbeat = round(structure_result.downbeat, 3)
        if structure_result.time_signature:
            time_signature = (
                f"{structure_result.time_signature[0]}/{structure_result.time_signature[1]}"
            )

        # Format structure spans with 1-indexed bars
        structure = []
        for s in structure_result.sections:
            if s.start_bar is not None and s.end_bar is not None:
                # Convert 0-indexed internal bars to 1-indexed output
                structure.append([int(s.start_bar) + 1, int(s.end_bar) + 1, s.label])
            else:
                structure.append([round(s.start_time, 1), round(s.end_time, 1), s.label])

        # Format events (already 1-indexed from analyze_structure)
        events = (
            [[bar, label] for bar, label in structure_result.events]
            if structure_result.events
            else None
        )

        # Format raw sections
        raw = (
            [
                {
                    "start": r.start,
                    "end": r.end,
                    "start_bar": r.start_bar,
                    "end_bar": r.end_bar,
                    "label": r.label,
                    "confidence": r.confidence,
                }
                for r in structure_result.raw
            ]
            if structure_result.raw
            else None
        )

    return TrackAnalysis(
        file=str(filepath),
        duration=duration,
        bpm=bpm,
        downbeat=downbeat,
        time_signature=time_signature,
        structure=structure,
        events=events,
        raw=raw,
    )


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
    structure_detector: str = "auto",
    annotations: bool = False,
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
        structure_detector: Structure detection method (auto, msaf, energy).
        annotations: Also output simplified .annotations.yaml templates.
    """
    # Load configuration
    load_config(config_path)

    # Determine which analyses to run
    run_bpm = analysis_types is None or "bpm" in analysis_types
    run_structure = (
        analysis_types is None or "structure" in analysis_types or "grid" in analysis_types
    )
    run_beats = analysis_types is not None and "beats" in analysis_types

    # MSAF is thread-safe and can run in parallel
    # No special handling needed unlike the previous allin1 detector

    # Collect all audio files
    audio_files = discover_audio_files(files, recursive=recursive)

    if not audio_files:
        logger.warning("no audio files found", paths=[str(f) for f in files])
        console.print("[yellow]No audio files found[/yellow]")
        return

    logger.info(
        "starting analysis",
        file_count=len(audio_files),
        run_bpm=run_bpm,
        run_structure=run_structure,
        run_beats=run_beats,
        offline=offline,
        workers=workers,
    )
    if not quiet:
        console.print(f"Found {len(audio_files)} file(s) to analyze")
        if workers > 1:
            console.print(f"Using {workers} parallel workers")

    # Prepare args for each file
    args_list = [
        (
            str(filepath),
            run_bpm,
            run_structure,
            run_beats,
            offline,
            ignore_metadata,
            structure_detector,
        )
        for filepath in audio_files
    ]

    # Process files
    start_time = time.time()

    # Always use parallel processing (workers=1 is handled by ParallelProcessor)
    results = _process_parallel(args_list, workers, console, quiet)

    total_time = time.time() - start_time

    # Display results
    if annotations:
        output_annotations(results, output, console, quiet)
    elif format == "yaml":
        output_yaml(results, output, console, quiet)
    elif format == "json" or output:
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


def analyze_file(
    filepath: Path,
    run_bpm: bool,
    run_structure: bool,
    offline: bool,
    ignore_metadata: bool,
    structure_detector: str = "auto",
    run_beats: bool = False,
) -> dict:
    """Analyze a single audio file.

    Args:
        filepath: Path to the audio file.
        run_bpm: Run BPM analysis.
        run_structure: Run structure analysis.
        offline: Skip network lookups.
        ignore_metadata: Skip metadata reading.
        structure_detector: Structure detection method (auto, msaf, energy).
        run_beats: Run beat detection.

    Returns:
        Analysis results in new schema format.
    """
    logger.info("analyzing file", filepath=str(filepath))
    analysis = _analyze_file_impl(
        filepath, run_bpm, run_structure, run_beats, offline, ignore_metadata, structure_detector
    )
    return analysis.to_dict()


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


def output_yaml(results: list[dict], output_path: Path | None, console: Console, quiet: bool):
    """Output results as YAML (multi-document for batch).

    Args:
        results: Analysis results.
        output_path: File to write YAML to (None for stdout).
        console: Rich console for output.
        quiet: Suppress output messages.
    """
    # Multi-document YAML: each track as separate document
    yaml_str = yaml.dump_all(
        results,
        default_flow_style=None,
        allow_unicode=True,
        sort_keys=False,
    )

    if output_path:
        output_path.write_text(yaml_str)
        if not quiet:
            console.print(f"Results written to {output_path}")
    else:
        console.print(yaml_str)


def output_annotations(
    results: list[dict], output_path: Path | None, console: Console, quiet: bool
):
    """Output results as simplified annotation templates.

    Args:
        results: Analysis results.
        output_path: File to write YAML to (None for stdout).
        console: Rich console for output.
        quiet: Suppress output messages.
    """
    output_lines = []

    for result in results:
        if "error" in result:
            continue

        # Extract timing info for timestamp calculation
        bpm = result.get("bpm")
        downbeat = result.get("downbeat", 0.0)

        # Build simplified annotation list from structure and events
        annotations = []

        # Add structure section starts
        if result.get("structure"):
            for section in result["structure"]:
                if len(section) >= 3:
                    start_bar, _, label = section[0], section[1], section[2]
                    # Calculate timestamp for this bar
                    if bpm:
                        time_seconds = bars_to_time(start_bar, bpm, first_downbeat=downbeat)
                        timestamp = (
                            format_time(time_seconds) if time_seconds is not None else "0:00.00s"
                        )
                    else:
                        timestamp = "0:00.00s"
                    annotations.append([start_bar, label, timestamp])

        # Add events
        if result.get("events"):
            for event in result["events"]:
                if len(event) >= 2:
                    bar, label = event[0], event[1]
                    # Calculate timestamp for this bar
                    if bpm:
                        time_seconds = bars_to_time(bar, bpm, first_downbeat=downbeat)
                        timestamp = (
                            format_time(time_seconds) if time_seconds is not None else "0:00.00s"
                        )
                    else:
                        timestamp = "0:00.00s"
                    annotations.append([bar, label, timestamp])

        # Sort by bar number
        annotations.sort(key=lambda x: x[0])

        # Build annotation document with formatted duration
        duration_sec = result.get("duration")
        duration_str = format_time(duration_sec) if duration_sec else None

        doc = {
            "file": result["file"],
            "duration": duration_str,
            "bpm": result.get("bpm"),
            "downbeat": result.get("downbeat"),
            "time_signature": result.get("time_signature"),
            "annotations": annotations,
        }

        # Dump the main document
        yaml_str = yaml.dump(
            doc,
            default_flow_style=None,
            allow_unicode=True,
            sort_keys=False,
        )
        output_lines.append(yaml_str)

        # Build commented raw events section with original detected times
        raw_lines = ["# --- Raw detected events (original analysis, do not edit) ---"]

        if result.get("raw"):
            for raw_item in result["raw"]:
                start_sec = raw_item.get("start", 0)
                end_sec = raw_item.get("end", 0)
                start_bar = raw_item.get("start_bar")
                end_bar = raw_item.get("end_bar")
                label = raw_item.get("label", "unknown")
                confidence = raw_item.get("confidence", 0)

                # Format times as MM:SS.ss
                start_fmt = format_time(start_sec)
                end_fmt = format_time(end_sec)

                # Build info string
                bar_info = ""
                if start_bar is not None and end_bar is not None:
                    bar_info = f" bars={start_bar:.2f}-{end_bar:.2f}"

                raw_lines.append(
                    f"# - {label}: {start_fmt} - {end_fmt}{bar_info} (conf={confidence:.2f})"
                )

        raw_lines.append("# --- End raw events ---")
        output_lines.append("\n".join(raw_lines))

    # Join documents with YAML document separator
    final_output = "\n---\n".join(output_lines)

    if output_path:
        output_path.write_text(final_output)
        if not quiet:
            console.print(f"Annotations written to {output_path}")
    else:
        console.print(final_output)


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
    table.add_column("Time Sig", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Sections", justify="right")

    for result in results:
        if "error" in result:
            table.add_row(
                Path(result["file"]).name,
                "[red]Error[/red]",
                "",
                "",
                "",
            )
        else:
            bpm = result.get("bpm", "N/A")
            time_sig = result.get("time_signature", "N/A")
            duration = result.get("duration", "N/A")
            duration_str = f"{duration}s" if duration != "N/A" else "N/A"
            sections = len(result.get("structure", [])) if result.get("structure") else "N/A"

            table.add_row(
                Path(result["file"]).name,
                str(bpm),
                str(time_sig),
                duration_str,
                str(sections),
            )

    console.print(table)
