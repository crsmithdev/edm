"""BPM evaluation logic."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from edm.analysis.bpm import analyze_bpm
from edm.evaluation.common import (
    calculate_accuracy_within_tolerance,
    calculate_error_distribution,
    calculate_mae,
    calculate_rmse,
    create_symlinks,
    discover_audio_files,
    get_git_branch,
    get_git_commit,
    identify_outliers,
    sample_full,
    sample_random,
    save_error_distribution_plot,
    save_results_json,
    save_results_markdown,
)
from edm.evaluation.reference import load_reference_auto
from edm.processing.parallel import ParallelProcessor

logger = structlog.get_logger(__name__)


# Worker function for parallel execution (must be top-level for pickling)
def _evaluate_file_worker(args: tuple) -> dict:
    """Worker function for parallel BPM evaluation.

    Args:
        args: Tuple of (filepath, ref_value).

    Returns:
        Evaluation result dict with file path, reference, computed, error, and timing.
    """
    filepath, ref_value = args

    # Convert string back to Path if needed (for pickling)
    if isinstance(filepath, str):
        filepath = Path(filepath)

    start_time = time.time()

    try:
        # Compute BPM using analysis module (force computation by ignoring metadata and going offline)
        bpm_result = analyze_bpm(filepath, ignore_metadata=True, offline=True)
        computed_value = bpm_result.bpm
        computation_time = time.time() - start_time

        error = computed_value - ref_value

        return {
            "file": str(filepath),
            "reference": ref_value,
            "computed": computed_value,
            "error": error,
            "success": True,
            "computation_time": computation_time,
            "error_message": None,
        }

    except Exception as e:
        computation_time = time.time() - start_time

        return {
            "file": str(filepath),
            "reference": ref_value,
            "computed": None,
            "error": None,
            "success": False,
            "computation_time": computation_time,
            "error_message": str(e),
        }


def _evaluate_sequential(args_list: list[tuple]) -> list[dict]:
    """Evaluate files sequentially.

    Args:
        args_list: List of (filepath, ref_value) tuples.

    Returns:
        List of evaluation results.
    """
    results = []

    for idx, args in enumerate(args_list, 1):
        filepath = args[0]
        if isinstance(filepath, str):
            filepath = Path(filepath)

        logger.debug(
            "evaluating file",
            progress=f"{idx}/{len(args_list)}",
            file=filepath.name,
        )

        result = _evaluate_file_worker(args)
        results.append(result)

        if result["success"]:
            logger.debug(
                "evaluation success",
                file=filepath.name,
                reference=result["reference"],
                computed=result["computed"],
                error=result["error"],
            )
        else:
            logger.error(
                "evaluation failed",
                file=filepath.name,
                error=result["error_message"],
            )

    return results


def _evaluate_parallel(args_list: list[tuple], workers: int) -> list[dict]:
    """Evaluate files in parallel.

    Args:
        args_list: List of (filepath, ref_value) tuples.
        workers: Number of parallel workers.

    Returns:
        List of evaluation results.
    """
    logger.info("starting parallel evaluation", workers=workers, files=len(args_list))

    completed = [0]  # Use list to allow mutation in callback

    def progress_callback(count: int):
        advance = count - completed[0]
        if advance > 0:
            completed[0] = count
            logger.debug(
                "parallel progress",
                completed=count,
                total=len(args_list),
            )

    processor = ParallelProcessor(
        worker_fn=_evaluate_file_worker,
        workers=workers,
        progress_callback=progress_callback,
    )

    try:
        results = processor.process(args_list)
    except KeyboardInterrupt:
        logger.warning("evaluation interrupted")
        raise

    # Log summary for each result
    for result in results:
        filepath = Path(result["file"])
        if result["success"]:
            logger.debug(
                "evaluation success",
                file=filepath.name,
                reference=result["reference"],
                computed=result["computed"],
                error=result["error"],
            )
        else:
            logger.error(
                "evaluation failed",
                file=filepath.name,
                error=result["error_message"],
            )

    return results


def evaluate_bpm(
    source_path: Path,
    reference_source: str,
    sample_size: int = 100,
    output_dir: Path | None = None,
    seed: int | None = None,
    full: bool = False,
    tolerance: float = 2.5,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate BPM detection accuracy.

    Args:
        source_path: Directory containing audio files
        reference_source: Reference source ('spotify', 'metadata', or file path)
        sample_size: Number of files to sample (ignored if full=True)
        output_dir: Output directory for results (default: data/accuracy/bpm/)
        seed: Random seed for reproducibility
        full: Use all files instead of sampling
        tolerance: BPM tolerance for accuracy calculation
        workers: Number of parallel workers (default: 1 = sequential)

    Returns:
        Dictionary containing evaluation results
    """
    logger.info(
        "starting bpm evaluation",
        source=str(source_path),
        reference=reference_source,
        sample_size=sample_size,
        full=full,
        tolerance=tolerance,
        workers=workers,
    )

    # Discover audio files
    all_files = discover_audio_files(source_path)
    if not all_files:
        raise ValueError(f"No audio files found in {source_path}")

    # Sample files
    if full:
        sampled_files = sample_full(all_files)
        sampling_strategy = "full"
    else:
        sampled_files = sample_random(all_files, sample_size, seed)
        sampling_strategy = "random"

    # Load reference data
    reference = load_reference_auto(
        reference_arg=reference_source,
        analysis_type="bpm",
        source_path=source_path,
        value_field="bpm",
    )

    if not reference:
        raise ValueError(f"No reference data loaded from {reference_source}")

    # Filter sampled files to only those with reference data
    sampled_files = [f for f in sampled_files if f.resolve() in reference]

    if not sampled_files:
        raise ValueError(
            f"No sampled files have reference data. "
            f"Sampled {len(sampled_files)} files but found 0 with reference."
        )

    logger.info(
        "evaluation setup complete",
        total_files=len(all_files),
        sampled=len(sampled_files),
        with_reference=len(sampled_files),
    )

    # Prepare args for each file
    args_list = [(str(file_path), reference[file_path.resolve()]) for file_path in sampled_files]

    # Process files (sequential or parallel)
    if workers == 1:
        results = _evaluate_sequential(args_list)
    else:
        results = _evaluate_parallel(args_list, workers)

    # Count successful and failed evaluations
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    # Calculate metrics
    errors = [r["error"] for r in results if r["success"]]

    if not errors:
        raise ValueError("No successful evaluations - cannot calculate metrics")

    mae = calculate_mae(errors)
    rmse = calculate_rmse(errors)
    accuracy = calculate_accuracy_within_tolerance(errors, tolerance)
    error_dist = calculate_error_distribution(errors)
    outliers = identify_outliers(results, n=10)

    # Prepare results dictionary
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()
    git_branch = get_git_branch()

    evaluation_results = {
        "metadata": {
            "analysis_type": "bpm",
            "timestamp": timestamp,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "sample_size": len(sampled_files),
            "sampling_strategy": sampling_strategy,
            "sampling_seed": seed,
            "reference_source": reference_source,
            "tolerance": tolerance,
        },
        "summary": {
            "total_files": len(sampled_files),
            "successful": successful,
            "failed": failed,
            "mean_absolute_error": mae,
            "root_mean_square_error": rmse,
            "accuracy_within_tolerance": accuracy,
            "error_distribution": error_dist,
        },
        "outliers": outliers,
        "results": results,
    }

    # Save results
    if output_dir is None:
        output_dir = Path("data/accuracy/bpm")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_base = output_dir / f"{timestamp_str}_bpm_eval_commit-{git_commit}"

    save_results_json(evaluation_results, output_base.with_suffix(".json"))
    save_results_markdown(evaluation_results, output_base.with_suffix(".md"))
    save_error_distribution_plot(errors, output_base.with_suffix(".png"), "BPM")

    # Create symlinks
    create_symlinks(output_base)

    logger.info(
        "evaluation complete", mae=mae, rmse=rmse, accuracy=accuracy, output=str(output_dir)
    )

    # Print summary to console
    print("\n" + "=" * 60)
    print("BPM EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total Files: {len(sampled_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print("Metrics:")
    print(f"  MAE: {mae:.2f} BPM")
    print(f"  RMSE: {rmse:.2f} BPM")
    print(f"  Accuracy (±{tolerance} BPM): {accuracy:.1f}%")
    print()
    print("Results saved to:")
    print(f"  - {output_base.with_suffix('.json')}")
    print(f"  - {output_base.with_suffix('.md')}")
    if output_base.with_suffix(".png").exists():
        print(f"  - {output_base.with_suffix('.png')}")
    print(f"  - {output_dir / 'latest.json'} (symlink)")
    print(f"  - {output_dir / 'latest.md'} (symlink)")
    print("=" * 60 + "\n")

    return evaluation_results
