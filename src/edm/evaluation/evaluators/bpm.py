"""BPM evaluation logic."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

logger = structlog.get_logger(__name__)


def evaluate_bpm(
    source_path: Path,
    reference_source: str,
    sample_size: int = 100,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    full: bool = False,
    tolerance: float = 2.5,
) -> Dict[str, Any]:
    """Evaluate BPM detection accuracy.

    Args:
        source_path: Directory containing audio files
        reference_source: Reference source ('spotify', 'metadata', or file path)
        sample_size: Number of files to sample (ignored if full=True)
        output_dir: Output directory for results (default: benchmarks/results/accuracy/bpm/)
        seed: Random seed for reproducibility
        full: Use all files instead of sampling
        tolerance: BPM tolerance for accuracy calculation

    Returns:
        Dictionary containing evaluation results
    """
    logger.info("starting_bpm_evaluation",
                source=str(source_path),
                reference=reference_source,
                sample_size=sample_size,
                full=full,
                tolerance=tolerance)

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
        value_field="bpm"
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

    logger.info("evaluation_setup_complete",
                total_files=len(all_files),
                sampled=len(sampled_files),
                with_reference=len(sampled_files))

    # Evaluate each file
    results = []
    successful = 0
    failed = 0

    for idx, file_path in enumerate(sampled_files, 1):
        ref_value = reference[file_path.resolve()]

        logger.info("evaluating_file",
                   progress=f"{idx}/{len(sampled_files)}",
                   file=file_path.name)

        start_time = time.time()

        try:
            # Compute BPM using analysis module (force computation by ignoring metadata and going offline)
            bpm_result = analyze_bpm(file_path, ignore_metadata=True, offline=True)
            computed_value = bpm_result.bpm
            computation_time = time.time() - start_time

            error = computed_value - ref_value

            results.append({
                "file": str(file_path),
                "reference": ref_value,
                "computed": computed_value,
                "error": error,
                "success": True,
                "computation_time": computation_time,
                "error_message": None,
            })

            successful += 1

            logger.info("evaluation_success",
                       file=file_path.name,
                       reference=ref_value,
                       computed=computed_value,
                       error=error)

        except Exception as e:
            computation_time = time.time() - start_time

            results.append({
                "file": str(file_path),
                "reference": ref_value,
                "computed": None,
                "error": None,
                "success": False,
                "computation_time": computation_time,
                "error_message": str(e),
            })

            failed += 1

            logger.error("evaluation_failed",
                        file=file_path.name,
                        error=str(e))

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
        output_dir = Path("benchmarks/results/accuracy/bpm")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_base = output_dir / f"{timestamp_str}_bpm_eval_commit-{git_commit}"

    save_results_json(evaluation_results, output_base.with_suffix(".json"))
    save_results_markdown(evaluation_results, output_base.with_suffix(".md"))
    save_error_distribution_plot(errors, output_base.with_suffix(".png"), "BPM")

    # Create symlinks
    create_symlinks(output_base)

    logger.info("evaluation_complete",
                mae=mae,
                rmse=rmse,
                accuracy=accuracy,
                output=str(output_dir))

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
    if output_base.with_suffix('.png').exists():
        print(f"  - {output_base.with_suffix('.png')}")
    print(f"  - {output_dir / 'latest.json'} (symlink)")
    print(f"  - {output_dir / 'latest.md'} (symlink)")
    print("=" * 60 + "\n")

    return evaluation_results
