#!/usr/bin/env python3
"""Benchmark different structure detectors against ground truth.

Compares ML, MSAF, and energy-based detectors on the same dataset.

Usage:
    python scripts/benchmark_detectors.py \\
        --reference data/ground_truth/structure.csv \\
        --audio data/audio/ \\
        --ml-model experiments/best_model.pt \\
        --output reports/benchmarks/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from edm.evaluation.evaluators.structure import evaluate_structure


def main():
    parser = argparse.ArgumentParser(description="Benchmark structure detectors")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="CSV file with ground truth annotations",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=None,
        help="Path to ML model checkpoint (for ML detector)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmarks"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of files to evaluate",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all files (ignore sample-size)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 3.0],
        help="Boundary tolerances in seconds",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["ml", "msaf", "energy"],
        help="Detectors to benchmark",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STRUCTURE DETECTOR BENCHMARK")
    print("=" * 80)
    print(f"Reference: {args.reference}")
    print(f"Audio: {args.audio}")
    print(f"Detectors: {', '.join(args.detectors)}")
    print(f"Sample: {'full' if args.full else f'{args.sample_size} files'}")
    print(f"Tolerances: {args.tolerances}")
    print("=" * 80)
    print()

    # Run evaluation for each detector
    results_by_detector = {}

    for detector in args.detectors:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {detector.upper()}")
        print("=" * 80)

        # Skip ML if no model provided
        if detector == "ml" and args.ml_model is None:
            print("⚠️  Skipping ML detector (no --ml-model provided)")
            continue

        # Create detector-specific output dir
        detector_output = args.output / detector
        detector_output.mkdir(parents=True, exist_ok=True)

        try:
            # Run evaluation at each tolerance
            detector_results = {}

            for tolerance in args.tolerances:
                print(f"\nTolerance: ±{tolerance}s")

                result = evaluate_structure(
                    source_path=args.audio,
                    reference_path=args.reference,
                    sample_size=args.sample_size,
                    output_dir=detector_output,
                    seed=args.seed,
                    full=args.full,
                    tolerance=tolerance,
                    detector=detector,
                )

                detector_results[tolerance] = {
                    "boundary_precision": result["summary"]["avg_boundary_precision"],
                    "boundary_recall": result["summary"]["avg_boundary_recall"],
                    "boundary_f1": result["summary"]["avg_boundary_f1"],
                    "event_precision": result["summary"]["avg_event_precision"],
                    "event_recall": result["summary"]["avg_event_recall"],
                    "successful": result["summary"]["successful"],
                    "failed": result["summary"]["failed"],
                }

            results_by_detector[detector] = detector_results

        except Exception as e:
            print(f"❌ Error evaluating {detector}: {e}")
            results_by_detector[detector] = {"error": str(e)}

    # Save comparative results
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary_path = args.output / f"{timestamp}_benchmark_summary.json"

    benchmark_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "reference": str(args.reference),
            "audio": str(args.audio),
            "sample_size": args.sample_size,
            "full": args.full,
            "seed": args.seed,
            "tolerances": args.tolerances,
            "detectors": args.detectors,
        },
        "results": results_by_detector,
    }

    with open(summary_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("BOUNDARY F1 COMPARISON")
    print("=" * 80)
    print(f"{'Detector':<15} ", end="")
    for tol in args.tolerances:
        print(f"±{tol}s      ", end="")
    print()
    print("-" * 80)

    for detector, results in results_by_detector.items():
        if "error" in results:
            print(f"{detector:<15} ERROR: {results['error']}")
            continue

        print(f"{detector:<15} ", end="")
        for tol in args.tolerances:
            if tol in results:
                f1 = results[tol]["boundary_f1"]
                print(f"{f1:>6.1%}   ", end="")
            else:
                print(f"{'N/A':>6}   ", end="")
        print()

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
