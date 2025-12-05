#!/usr/bin/env python3
"""Ablation study for ML structure detection.

Tests impact of:
- Different backbones (MERT vs CNN)
- Different prediction heads (boundary, energy, beat, label)
- Training data size

Usage:
    python scripts/ablation_study.py \\
        --models experiments/ablation/*.pt \\
        --reference data/ground_truth/structure.csv \\
        --audio data/audio/ \\
        --output reports/ablation/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from edm.evaluation.evaluators.structure import evaluate_structure


def parse_model_config(model_path: Path) -> dict[str, str]:
    """Parse model configuration from filename.

    Expected format: {backbone}_{heads}_{datasize}.pt
    Example: mert_boundary+energy_1000.pt

    Args:
        model_path: Path to model checkpoint

    Returns:
        Dict with backbone, heads, data_size
    """
    stem = model_path.stem
    parts = stem.split("_")

    if len(parts) >= 3:
        return {
            "backbone": parts[0],
            "heads": parts[1],
            "data_size": parts[2],
        }
    else:
        # Fallback: use filename as identifier
        return {
            "backbone": "unknown",
            "heads": "unknown",
            "data_size": "unknown",
            "identifier": stem,
        }


def main():
    parser = argparse.ArgumentParser(description="Ablation study for ML detector")
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        required=True,
        help="Model checkpoints to evaluate",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="CSV file with ground truth",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Audio directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablation"),
        help="Output directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of files to evaluate per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Boundary tolerance in seconds",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABLATION STUDY: ML STRUCTURE DETECTOR")
    print("=" * 80)
    print(f"Models: {len(args.models)}")
    print(f"Sample size: {args.sample_size} files")
    print(f"Tolerance: ±{args.tolerance}s")
    print("=" * 80)
    print()

    results = []

    for model_path in args.models:
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {model_path.name}")
        print("=" * 80)

        # Parse model config
        config = parse_model_config(model_path)
        print(f"Backbone: {config.get('backbone', 'unknown')}")
        print(f"Heads: {config.get('heads', 'unknown')}")
        print(f"Data size: {config.get('data_size', 'unknown')}")

        try:
            # Run evaluation
            eval_result = evaluate_structure(
                source_path=args.audio,
                reference_path=args.reference,
                sample_size=args.sample_size,
                output_dir=args.output / model_path.stem,
                seed=args.seed,
                full=False,
                tolerance=args.tolerance,
                detector="ml",
            )

            result = {
                "model": str(model_path),
                "config": config,
                "metrics": {
                    "boundary_precision": eval_result["summary"]["avg_boundary_precision"],
                    "boundary_recall": eval_result["summary"]["avg_boundary_recall"],
                    "boundary_f1": eval_result["summary"]["avg_boundary_f1"],
                    "event_precision": eval_result["summary"]["avg_event_precision"],
                    "event_recall": eval_result["summary"]["avg_event_recall"],
                },
                "summary": {
                    "successful": eval_result["summary"]["successful"],
                    "failed": eval_result["summary"]["failed"],
                },
            }

            results.append(result)

            print(f"\nBoundary F1: {result['metrics']['boundary_f1']:.1%}")
            print(f"Event Precision: {result['metrics']['event_precision']:.1%}")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {
                    "model": str(model_path),
                    "config": config,
                    "error": str(e),
                }
            )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = args.output / f"{timestamp}_ablation_results.json"

    ablation_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "reference": str(args.reference),
            "audio": str(args.audio),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "tolerance": args.tolerance,
        },
        "results": results,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Boundary F1':<15} {'Event Prec':<15}")
    print("-" * 80)

    for result in results:
        model_name = Path(result["model"]).stem[:38]
        if "error" in result:
            print(f"{model_name:<40} {'ERROR':<15}")
        else:
            f1 = result["metrics"]["boundary_f1"]
            event_p = result["metrics"]["event_precision"]
            print(f"{model_name:<40} {f1:<14.1%} {event_p:<14.1%}")

    print("=" * 80)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
