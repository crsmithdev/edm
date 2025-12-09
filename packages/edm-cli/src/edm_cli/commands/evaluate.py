"""Evaluate command - accuracy evaluation against reference annotations."""

import json
from datetime import datetime
from pathlib import Path

import typer
import yaml
from edm.evaluation.common import get_git_branch, get_git_commit
from edm.evaluation.evaluators.structure import _evaluate_file_worker
from edm.processing import ParallelProcessor, get_default_workers


def evaluate_command(
    reference: Path = typer.Option(
        Path("data/annotations"),
        "--reference",
        "-r",
        help="Directory containing reference annotation YAML files",
        file_okay=False,
        dir_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
        file_okay=False,
        dir_okay=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress detailed output",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (default: CPU count - 1)",
    ),
):
    """Evaluate analysis accuracy against reference annotations.

    Loads annotation files from data/annotations/ (or specified dir),
    runs analysis on each audio file, and compares results.

    Uses fixed defaults: tolerance=2.0s, detector=auto (msaf).

    Skips files without corresponding annotations or missing audio files.

    Examples:

        edm evaluate

        edm evaluate --reference data/annotations
    """
    # Fixed defaults
    tolerance = 2.0
    detector = "auto"
    if not reference.exists():
        typer.echo(f"Error: Reference directory not found: {reference}", err=True)
        raise typer.Exit(code=1)

    # Load reference annotations
    annotations = load_yaml_annotations(reference)

    if not annotations:
        typer.echo(f"Error: No valid annotation files found in {reference}", err=True)
        raise typer.Exit(code=1)

    # Filter to existing files and build args list
    args_list = []
    for audio_path, ref_sections in annotations.items():
        if not audio_path.exists():
            if not quiet:
                typer.echo(f"  Skipping (file not found): {audio_path.name}")
            continue
        args_list.append((str(audio_path), ref_sections, tolerance, detector))

    if not args_list:
        typer.echo("Error: No valid files to evaluate", err=True)
        raise typer.Exit(code=1)

    # Auto-determine workers if not specified
    if workers is None:
        workers = min(get_default_workers(), len(args_list))

    if not quiet:
        typer.echo(f"Loaded {len(annotations)} annotation file(s) from {reference}")
        if workers > 1:
            typer.echo(f"Using {workers} parallel workers")

    # Evaluate files in parallel
    processor = ParallelProcessor(worker_fn=_evaluate_file_worker, workers=workers)
    results = processor.process(args_list)

    # Count successes and failures
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    # Print individual results if not quiet
    if not quiet:
        for result in results:
            audio_path = Path(result["file"])
            typer.echo(f"  Evaluated: {audio_path.name}")
            if result["success"]:
                typer.echo(
                    f"    boundary_f1={result['boundary_f1']:.1%} event_f1={result['event_f1']:.1%}"
                )
            else:
                typer.echo(f"    ERROR: {result['error_message']}")

    if not results:
        typer.echo("Error: No files evaluated", err=True)
        raise typer.Exit(code=1)

    # Calculate aggregate metrics
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        typer.echo("Error: No successful evaluations", err=True)
        raise typer.Exit(code=1)

    avg_boundary_precision = sum(r["boundary_precision"] for r in successful_results) / len(
        successful_results
    )
    avg_boundary_recall = sum(r["boundary_recall"] for r in successful_results) / len(
        successful_results
    )
    avg_boundary_f1 = sum(r["boundary_f1"] for r in successful_results) / len(successful_results)
    avg_event_precision = sum(r["event_precision"] for r in successful_results) / len(
        successful_results
    )
    avg_event_recall = sum(r["event_recall"] for r in successful_results) / len(successful_results)
    avg_event_f1 = sum(r["event_f1"] for r in successful_results) / len(successful_results)

    # Print summary
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("EVALUATION COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"Total Files: {len(results)}")
    typer.echo(f"Successful: {successful}")
    typer.echo(f"Failed: {failed}")
    typer.echo("")
    typer.echo("Boundary Detection:")
    typer.echo(f"  Precision: {avg_boundary_precision:.1%}")
    typer.echo(f"  Recall: {avg_boundary_recall:.1%}")
    typer.echo(f"  F1: {avg_boundary_f1:.1%}")
    typer.echo("")
    typer.echo("Event Detection:")
    typer.echo(f"  Precision: {avg_event_precision:.1%}")
    typer.echo(f"  Recall: {avg_event_recall:.1%}")
    typer.echo(f"  F1: {avg_event_f1:.1%}")
    typer.echo("=" * 60)

    # Save results if output specified
    if output:
        output.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        git_commit = get_git_commit()

        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "git_commit": git_commit,
                "git_branch": get_git_branch(),
                "reference_dir": str(reference),
                "tolerance": tolerance,
                "detector": detector,
            },
            "summary": {
                "total_files": len(results),
                "successful": successful,
                "failed": failed,
                "avg_boundary_precision": avg_boundary_precision,
                "avg_boundary_recall": avg_boundary_recall,
                "avg_boundary_f1": avg_boundary_f1,
                "avg_event_precision": avg_event_precision,
                "avg_event_recall": avg_event_recall,
                "avg_event_f1": avg_event_f1,
            },
            "results": results,
        }

        output_file = output / f"{timestamp}_eval_commit-{git_commit}.json"
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        typer.echo(f"\nResults saved to: {output_file}")


def load_yaml_annotations(annotations_dir: Path) -> dict[Path, list[dict]]:
    """Load reference annotations from YAML files in a directory.

    Supports both old schema (file/bpm/annotations at root) and new schema
    (audio/structure sections with metadata).

    Args:
        annotations_dir: Directory containing .yaml annotation files.

    Returns:
        Dictionary mapping audio file paths to lists of annotation dicts.
    """
    from edm.analysis.bars import bars_to_time

    reference: dict[Path, list[dict]] = {}

    for yaml_path in annotations_dir.glob("*.yaml"):
        try:
            with open(yaml_path) as f:
                # Only read first YAML document (ignore commented raw events section)
                # Use safe_load_all to handle multi-document YAML, take first doc only
                docs = list(yaml.safe_load_all(f))
                doc = docs[0] if docs else None

            if not doc:
                continue

            # Detect schema: new schema has 'audio' and 'structure', old has 'file'
            if "audio" in doc and "structure" in doc:
                # New schema
                audio_path = Path(doc["audio"]["file"]).resolve()
                bpm = doc["audio"].get("bpm")
                downbeat = doc["audio"].get("downbeat", 0.0)
                annotations = doc["structure"]
            elif "file" in doc:
                # Old schema (backward compatibility)
                audio_path = Path(doc["file"]).resolve()
                bpm = doc.get("bpm")
                downbeat = doc.get("downbeat", 0.0)
                annotations = doc.get("annotations", [])
            else:
                continue

            if not annotations or not bpm:
                continue

            sections = []
            prev_label = None
            for i, ann in enumerate(annotations):
                # Handle both old schema (list/tuple) and new schema (dict)
                if isinstance(ann, dict):
                    # New schema: {"bar": 1, "label": "intro", "time": 0.907}
                    bar = ann.get("bar")
                    label = ann.get("label", "").strip().lower()
                elif isinstance(ann, (list, tuple)) and len(ann) >= 2:
                    # Old schema: [1, "intro"] or (1, "intro")
                    bar, label = ann[0], ann[1]
                    label = label.strip().lower()
                else:
                    continue

                if not bar or not label:
                    continue

                # Skip kick in/kick out events - they mark kick drum presence, not structure
                if label in {"kick in", "kick out"}:
                    continue

                # Convert bar to time
                time_val = bars_to_time(bar, bpm, first_downbeat=downbeat)
                if time_val is None:
                    continue

                # Determine end time (next non-kick annotation's start, or track end)
                end_time = None
                for j in range(i + 1, len(annotations)):
                    # Handle both schemas when looking ahead
                    if isinstance(annotations[j], dict):
                        next_label = annotations[j].get("label", "").strip().lower()
                        next_bar = annotations[j].get("bar")
                    elif isinstance(annotations[j], (list, tuple)) and len(annotations[j]) >= 2:
                        next_label = annotations[j][1].strip().lower()
                        next_bar = annotations[j][0]
                    else:
                        continue

                    if next_label not in {"kick in", "kick out"} and next_bar:
                        end_time = bars_to_time(next_bar, bpm, first_downbeat=downbeat)
                        break

                # Normalize labels:
                # - 'other' after 'breakdown' implies a drop
                # - 'build' is equivalent to 'buildup'
                normalized_label = label
                if label == "other" and prev_label == "breakdown":
                    normalized_label = "drop"
                elif label == "build":
                    normalized_label = "buildup"

                # Check if this is an event label (drops are detected as events by our system)
                is_event = normalized_label == "drop"

                if is_event:
                    sections.append(
                        {
                            "label": normalized_label,
                            "time": time_val,
                            "is_event": True,
                        }
                    )
                elif end_time is not None:
                    sections.append(
                        {
                            "label": normalized_label,
                            "start": time_val,
                            "end": end_time,
                            "is_event": False,
                        }
                    )

                prev_label = label

            if sections:
                reference[audio_path] = sections

        except Exception as e:
            typer.echo(f"Warning: Failed to load {yaml_path}: {e}", err=True)
            continue

    return reference
