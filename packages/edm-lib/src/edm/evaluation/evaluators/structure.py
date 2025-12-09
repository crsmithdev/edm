"""Structure evaluation logic."""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from edm.analysis.bars import bars_to_time
from edm.analysis.structure import analyze_structure
from edm.evaluation.common import (
    create_symlinks,
    discover_audio_files,
    get_git_branch,
    get_git_commit,
    sample_full,
    sample_random,
    save_results_json,
)

logger = structlog.get_logger(__name__)


EVENT_LABELS = {"drop"}


def load_structure_reference(reference_path: Path) -> dict[Path, list[dict]]:
    """Load structure ground truth from CSV file.

    Expected CSV formats:

    Time-based spans:
        filename,start,end,label
        track1.mp3,0.0,32.0,intro
        track1.mp3,32.0,64.0,buildup
        ...

    Time-based events (no end column or end equals start):
        filename,start,label
        track1.mp3,32.0,drop
        ...

    Bar-based spans (requires BPM column):
        filename,start_bar,end_bar,label,bpm
        track1.mp3,0,16,intro,128
        track1.mp3,16,32,buildup,128
        ...

    Bar-based events (no end_bar or end_bar equals start_bar):
        filename,bar,label,bpm
        track1.mp3,17,drop,128
        ...

    Bar-based with first_downbeat (for accurate bar alignment):
        filename,start_bar,end_bar,label,bpm,first_downbeat
        track1.mp3,1,17,intro,128,0.5
        track1.mp3,17,33,buildup,128,0.5
        ...

    Args:
        reference_path: Path to CSV file with structure annotations.

    Returns:
        Dictionary mapping file paths to lists of section dicts.
        Spans have 'start' and 'end' in seconds.
        Events have 'time' in seconds and 'is_event': True.
    """
    reference: dict[Path, list[dict]] = {}

    with open(reference_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column names for filename
            filename = row.get("filename") or row.get("file")
            if not filename:
                logger.warning("skipping row without filename", row=row)
                continue

            filepath = Path(filename).resolve()
            label = row["label"].strip().lower()

            # Check if this is an event (single point) or span
            is_event = label in EVENT_LABELS

            # Determine if we have bar-based or time-based annotations
            has_bar = "bar" in row and row["bar"]
            has_bars = "start_bar" in row and row["start_bar"]
            has_time = "start" in row and row["start"]
            bpm = float(row["bpm"]) if "bpm" in row and row["bpm"] else None
            first_downbeat = (
                float(row["first_downbeat"])
                if "first_downbeat" in row and row["first_downbeat"]
                else 0.0
            )

            if is_event:
                # Handle event (single point)
                if has_bar and bpm:
                    bar = float(row["bar"])
                    time_val = bars_to_time(bar, bpm, first_downbeat=first_downbeat)
                    if time_val is None:
                        logger.warning("failed to convert bar to time", filename=filename, bar=bar)
                        continue
                elif has_bars and bpm:
                    bar = float(row["start_bar"])
                    time_val = bars_to_time(bar, bpm, first_downbeat=first_downbeat)
                    if time_val is None:
                        logger.warning("failed to convert bar to time", filename=filename, bar=bar)
                        continue
                elif has_time:
                    time_val = float(row["start"])
                else:
                    logger.warning("event missing time/bar data", filename=filename, label=label)
                    continue

                section = {"label": label, "time": time_val, "is_event": True}
            else:
                # Handle span (start/end)
                has_end_bar = "end_bar" in row and row["end_bar"]
                has_end = "end" in row and row["end"]

                if has_bars and has_end_bar and bpm:
                    start_bar = float(row["start_bar"])
                    end_bar = float(row["end_bar"])
                    start_time = bars_to_time(start_bar, bpm, first_downbeat=first_downbeat)
                    end_time = bars_to_time(end_bar, bpm, first_downbeat=first_downbeat)

                    if start_time is None or end_time is None:
                        logger.warning(
                            "failed to convert bars to time",
                            filename=filename,
                            start_bar=start_bar,
                            end_bar=end_bar,
                        )
                        continue
                elif has_time and has_end:
                    start_time = float(row["start"])
                    end_time = float(row["end"])
                else:
                    logger.warning("span missing time/bar data", filename=filename, label=label)
                    continue

                section = {"label": label, "start": start_time, "end": end_time, "is_event": False}

            if filepath not in reference:
                reference[filepath] = []
            reference[filepath].append(section)

    # Sort sections by start time/time for each file
    for filepath in reference:
        reference[filepath] = sorted(
            reference[filepath], key=lambda s: s.get("start", s.get("time", 0))
        )

    logger.info("loaded structure reference", files=len(reference), path=str(reference_path))

    return reference


def _evaluate_file_worker(args: tuple) -> dict:
    """Worker function for parallel structure evaluation.

    Args:
        args: Tuple of (filepath, ref_sections, tolerance, detector).

    Returns:
        Evaluation result dict.
    """
    filepath, ref_sections, tolerance, detector = args

    if isinstance(filepath, str):
        filepath = Path(filepath)

    start_time = time.time()

    try:
        result = analyze_structure(filepath, detector=detector)
        computation_time = time.time() - start_time

        # Convert detected sections to comparable format
        detected = []

        # Add span sections (non-events)
        for s in result.sections:
            detected.append(
                {
                    "label": s.label,
                    "start": s.start_time,
                    "end": s.end_time,
                    "is_event": False,
                    "confidence": s.confidence,
                }
            )

        # Add events (drops, etc.) from result.events
        # Events are stored as (bar_number, label) tuples - convert to time
        if result.events and result.bpm:
            for bar, label in result.events:
                event_time = bars_to_time(bar, result.bpm, result.time_signature)
                if event_time is not None:
                    detected.append(
                        {
                            "label": label,
                            "time": event_time,
                            "is_event": True,
                            "confidence": 0.9,  # Events from detector have high confidence
                        }
                    )

        # Calculate metrics
        metrics = _calculate_structure_metrics(ref_sections, detected, tolerance)

        return {
            "file": str(filepath),
            "reference_sections": ref_sections,
            "detected_sections": detected,
            "detector": result.detector,
            "success": True,
            "computation_time": computation_time,
            "error_message": None,
            **metrics,
        }

    except Exception as e:
        computation_time = time.time() - start_time

        return {
            "file": str(filepath),
            "reference_sections": ref_sections,
            "detected_sections": [],
            "detector": None,
            "success": False,
            "computation_time": computation_time,
            "error_message": str(e),
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
        }


def _calculate_structure_metrics(
    reference: list[dict], detected: list[dict], tolerance: float
) -> dict:
    """Calculate structure detection metrics.

    Args:
        reference: Reference sections (spans and events).
        detected: Detected sections (spans and events).
        tolerance: Boundary tolerance in seconds.

    Returns:
        Dictionary of metrics.
    """
    if not reference or not detected:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
        }

    # Separate events and spans
    ref_events = [s for s in reference if s.get("is_event")]
    ref_spans = [s for s in reference if not s.get("is_event")]
    det_events = [s for s in detected if s.get("is_event")]
    det_spans = [s for s in detected if not s.get("is_event")]

    # Extract boundaries from spans (excluding 0.0 which is always present)
    ref_boundaries = set()
    for s in ref_spans:
        if s["start"] > 0.1:
            ref_boundaries.add(s["start"])
        if s["end"] > 0.1:
            ref_boundaries.add(s["end"])

    det_boundaries = set()
    for s in det_spans:
        if s["start"] > 0.1:
            det_boundaries.add(s["start"])
        if s["end"] > 0.1:
            det_boundaries.add(s["end"])

    # Match boundaries within tolerance
    matched_ref = set()
    matched_det = set()

    for ref_b in ref_boundaries:
        for det_b in det_boundaries:
            if abs(ref_b - det_b) <= tolerance and det_b not in matched_det:
                matched_ref.add(ref_b)
                matched_det.add(det_b)
                break

    # Calculate boundary metrics
    true_positives = len(matched_det)
    false_positives = len(det_boundaries) - true_positives
    false_negatives = len(ref_boundaries) - len(matched_ref)

    boundary_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    boundary_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if (boundary_precision + boundary_recall) > 0
        else 0.0
    )

    # Calculate event metrics (e.g., drops)
    matched_ref_events = set()
    matched_det_events = set()

    for i, ref_e in enumerate(ref_events):
        ref_time = ref_e["time"]
        for j, det_e in enumerate(det_events):
            det_time = det_e["time"]
            if (
                abs(ref_time - det_time) <= tolerance
                and j not in matched_det_events
                and _labels_match(ref_e["label"], det_e["label"])
            ):
                matched_ref_events.add(i)
                matched_det_events.add(j)
                break

    event_tp = len(matched_det_events)
    event_fp = len(det_events) - event_tp
    event_fn = len(ref_events) - len(matched_ref_events)

    event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0.0
    event_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0
        else 0.0
    )

    return {
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
    }


def _labels_match(ref_label: str, det_label: str) -> bool:
    """Check if reference and detected event labels match.

    Args:
        ref_label: Reference event label.
        det_label: Detected event label.

    Returns:
        True if labels match exactly.
    """
    return ref_label == det_label


def _calculate_overlap(section1: dict, section2: dict) -> float:
    """Calculate overlap ratio between two sections.

    Args:
        section1: First section with 'start' and 'end'.
        section2: Second section with 'start' and 'end'.

    Returns:
        Overlap ratio (0 to 1).
    """
    start = max(section1["start"], section2["start"])
    end = min(section1["end"], section2["end"])

    if end <= start:
        return 0.0

    overlap = end - start
    duration1 = section1["end"] - section1["start"]
    duration2 = section2["end"] - section2["start"]

    # Use IoU (intersection over union)
    union = duration1 + duration2 - overlap
    return overlap / union if union > 0 else 0.0


def evaluate_structure(
    source_path: Path,
    reference_path: Path,
    sample_size: int = 100,
    output_dir: Path | None = None,
    seed: int | None = None,
    full: bool = False,
    tolerance: float = 2.0,
    detector: str = "auto",
) -> dict[str, Any]:
    """Evaluate structure detection accuracy.

    Args:
        source_path: Directory containing audio files.
        reference_path: Path to CSV file with ground truth annotations.
        sample_size: Number of files to sample (ignored if full=True).
        output_dir: Output directory for results.
        seed: Random seed for reproducibility.
        full: Use all files instead of sampling.
        tolerance: Boundary tolerance in seconds.
        detector: Structure detector to use (auto, msaf, energy).

    Returns:
        Dictionary containing evaluation results.
    """
    logger.info(
        "starting structure evaluation",
        source=str(source_path),
        reference=str(reference_path),
        sample_size=sample_size,
        full=full,
        tolerance=tolerance,
        detector=detector,
    )

    # Discover audio files
    all_files = discover_audio_files(source_path)
    if not all_files:
        raise ValueError(f"No audio files found in {source_path}")

    # Load reference data
    reference = load_structure_reference(reference_path)
    if not reference:
        raise ValueError(f"No reference data loaded from {reference_path}")

    # Sample files
    if full:
        sampled_files = sample_full(all_files)
        sampling_strategy = "full"
    else:
        sampled_files = sample_random(all_files, sample_size, seed)
        sampling_strategy = "random"

    # Filter to files with reference data
    sampled_files = [f for f in sampled_files if f.resolve() in reference]

    if not sampled_files:
        raise ValueError(
            f"No sampled files have reference data. "
            f"Sampled {len(all_files)} files but found 0 with reference."
        )

    logger.info(
        "evaluation setup complete",
        total_files=len(all_files),
        sampled=len(sampled_files),
        with_reference=len(sampled_files),
    )

    # Evaluate each file
    results = []
    for idx, filepath in enumerate(sampled_files, 1):
        logger.debug(
            "evaluating file",
            progress=f"{idx}/{len(sampled_files)}",
            file=filepath.name,
        )

        ref_sections = reference[filepath.resolve()]
        result = _evaluate_file_worker((str(filepath), ref_sections, tolerance, detector))
        results.append(result)

        if result["success"]:
            logger.debug(
                "evaluation success",
                file=filepath.name,
                boundary_f1=result["boundary_f1"],
            )
        else:
            logger.error(
                "evaluation failed",
                file=filepath.name,
                error=result["error_message"],
            )

    # Count successes/failures
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    # Calculate aggregate metrics
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        raise ValueError("No successful evaluations - cannot calculate metrics")

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

    # Prepare results dictionary
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()
    git_branch = get_git_branch()

    evaluation_results = {
        "metadata": {
            "analysis_type": "structure",
            "timestamp": timestamp,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "sample_size": len(sampled_files),
            "sampling_strategy": sampling_strategy,
            "sampling_seed": seed,
            "reference_source": str(reference_path),
            "tolerance": tolerance,
            "detector": detector,
        },
        "summary": {
            "total_files": len(sampled_files),
            "successful": successful,
            "failed": failed,
            "avg_boundary_precision": avg_boundary_precision,
            "avg_boundary_recall": avg_boundary_recall,
            "avg_boundary_f1": avg_boundary_f1,
            "avg_event_precision": avg_event_precision,
            "avg_event_recall": avg_event_recall,
        },
        "results": results,
    }

    # Save results
    if output_dir is None:
        output_dir = Path("data/accuracy/structure")

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_base = output_dir / f"{timestamp_str}_structure_eval_commit-{git_commit}"

    save_results_json(evaluation_results, output_base.with_suffix(".json"))
    _save_structure_markdown(evaluation_results, output_base.with_suffix(".md"))

    create_symlinks(output_base)

    logger.info(
        "evaluation complete",
        boundary_f1=avg_boundary_f1,
        event_precision=avg_event_precision,
        event_recall=avg_event_recall,
        output=str(output_dir),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("STRUCTURE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total Files: {len(sampled_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print("Boundary Detection:")
    print(f"  Precision: {avg_boundary_precision:.1%}")
    print(f"  Recall: {avg_boundary_recall:.1%}")
    print(f"  F1: {avg_boundary_f1:.1%}")
    print()
    print("Event Detection:")
    print(f"  Precision: {avg_event_precision:.1%}")
    print(f"  Recall: {avg_event_recall:.1%}")
    print()
    print("Results saved to:")
    print(f"  - {output_base.with_suffix('.json')}")
    print(f"  - {output_base.with_suffix('.md')}")
    print(f"  - {output_dir / 'latest.json'} (symlink)")
    print("=" * 60 + "\n")

    return evaluation_results


def _save_structure_markdown(results: dict, output_path: Path) -> None:
    """Save structure evaluation results to Markdown.

    Args:
        results: Evaluation results dictionary.
        output_path: Output file path.
    """
    metadata = results["metadata"]
    summary = results["summary"]

    lines = [
        "# Structure Evaluation Results",
        "",
        f"**Date**: {metadata['timestamp']}",
        f"**Commit**: {metadata['git_commit']}",
        f"**Detector**: {metadata['detector']}",
        f"**Sample**: {metadata['sample_size']} files ({metadata['sampling_strategy']})",
        f"**Tolerance**: ±{metadata['tolerance']} seconds",
        "",
        "## Summary Metrics",
        "",
        "### Boundary Detection",
        f"- Precision: {summary['avg_boundary_precision']:.1%}",
        f"- Recall: {summary['avg_boundary_recall']:.1%}",
        f"- F1: {summary['avg_boundary_f1']:.1%}",
        "",
        "### Event Detection",
        f"- Precision: {summary['avg_event_precision']:.1%}",
        f"- Recall: {summary['avg_event_recall']:.1%}",
        "",
        "## Evaluation Summary",
        f"- Successful: {summary['successful']} / {summary['total_files']}",
        f"- Failed: {summary['failed']}",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("saved results markdown", path=str(output_path))
