#!/usr/bin/env python3
"""Detect label errors in training data using cleanlab.

This script:
1. Loads a trained model
2. Generates cross-validation predictions
3. Runs cleanlab error detection
4. Generates human-readable reports

Usage:
    python scripts/detect_label_errors.py \
        --checkpoint experiments/run_20250101_120000/checkpoints/best.pt \
        --annotations data/annotations/ \
        --audio data/audio/ \
        --output reports/label_errors/
"""

import argparse
from pathlib import Path

import torch

from edm.models.multitask import MultiTaskModel
from edm.training.cleanlab_utils import (
    cross_val_predictions,
    detect_label_errors,
    generate_error_report,
    save_error_metadata,
)
from edm.training.dataset import EDMDataset


def main():
    parser = argparse.ArgumentParser(description="Detect label errors using cleanlab")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory with annotation files",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Audio files directory (if different from annotation paths)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/label_errors"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Error detection threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = MultiTaskModel.load(args.checkpoint, device=device)
    print("Model loaded successfully")

    # Load dataset
    print(f"\nLoading dataset from {args.annotations}...")
    dataset = EDMDataset(
        annotation_dir=args.annotations,
        audio_dir=args.audio,
        duration=None,  # Full tracks for better error detection
        augment=False,
    )
    print(f"Loaded {len(dataset)} samples")

    # Generate cross-validation predictions
    print(f"\nGenerating {args.n_splits}-fold cross-validation predictions...")
    predictions = cross_val_predictions(
        model=model,
        dataset=dataset,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        device=device,
    )
    print("Predictions complete")

    # Detect errors
    print(f"\nDetecting label errors (threshold={args.threshold})...")
    error_analysis = detect_label_errors(
        predictions=predictions,
        threshold=args.threshold,
    )

    # Print summary
    if "label_summary" in error_analysis:
        summary = error_analysis["label_summary"]
        print("\nLabel errors:")
        print(f"  Total frames: {summary['total_frames']}")
        print(f"  Flagged issues: {summary['num_issues']}")
        print(f"  Error rate: {summary['error_rate']:.2%}")

    if "boundary_summary" in error_analysis:
        summary = error_analysis["boundary_summary"]
        print("\nBoundary errors:")
        print(f"  Total frames: {summary['total_frames']}")
        print(f"  Flagged issues: {summary['num_issues']}")
        print(f"  Error rate: {summary['error_rate']:.2%}")

    # Generate reports
    print(f"\nGenerating reports in {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)

    # Text report
    report_path = args.output / "error_report.txt"
    generate_error_report(
        dataset=dataset,
        error_analysis=error_analysis,
        output_path=report_path,
        top_k=50,
    )

    # Per-sample metadata
    metadata_dir = args.output / "metadata"
    save_error_metadata(
        dataset=dataset,
        error_analysis=error_analysis,
        output_dir=metadata_dir,
    )

    print("\nDone! Review the reports to identify problematic annotations.")


if __name__ == "__main__":
    main()
