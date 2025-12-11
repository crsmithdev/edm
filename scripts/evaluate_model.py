#!/usr/bin/env python3
"""Evaluate trained model checkpoint on validation set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from edm.evaluation.metrics import boundary_f1_at_tolerances
from edm.models.multitask import create_model
from edm.training.dataset import create_dataloaders


def load_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model and config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")

    # Load model state
    # We need to recreate the model architecture
    # This is a simplified version - adjust based on your config
    model = create_model(
        backbone_type="mert-95m",
        enable_boundary=True,
        enable_energy=True,
        enable_beat=True,
        enable_label=False,
        num_classes=6,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def extract_boundary_times(boundary_probs: np.ndarray, threshold: float = 0.5) -> list[float]:
    """Extract boundary times from probability predictions.

    Args:
        boundary_probs: Boundary probabilities [time, 1]
        threshold: Detection threshold

    Returns:
        List of boundary times in seconds (assuming 10 FPS frame rate)
    """
    # Apply threshold
    detections = boundary_probs[:, 0] > threshold

    # Find peaks (local maxima above threshold)
    boundaries = []
    for i in range(1, len(detections) - 1):
        if (
            detections[i]
            and boundary_probs[i, 0] > boundary_probs[i - 1, 0]
            and boundary_probs[i, 0] > boundary_probs[i + 1, 0]
        ):
            # Convert frame index to seconds (assuming 10 FPS)
            time_sec = i * 0.1
            boundaries.append(time_sec)

    return boundaries


def evaluate_model(
    checkpoint_path: Path,
    annotation_dir: Path,
    audio_dir: Path | None = None,
    batch_size: int = 4,
    device: str = "cuda",
    output_file: Path | None = None,
) -> dict:
    """Evaluate model on validation set.

    Args:
        checkpoint_path: Path to model checkpoint
        annotation_dir: Directory with annotations
        audio_dir: Directory with audio files
        batch_size: Batch size for evaluation
        device: Device to use
        output_file: Optional path to save results

    Returns:
        Dict with evaluation metrics
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    model, config = load_checkpoint(checkpoint_path, device)

    print(f"Creating dataloaders from {annotation_dir}")
    train_loader, val_loader = create_dataloaders(
        annotation_dir=annotation_dir,
        audio_dir=audio_dir,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=4,
        duration=30.0,
    )

    print(f"Evaluating on {len(val_loader)} validation batches")

    # Collect predictions and ground truth
    all_boundary_preds = []
    all_boundary_targets = []
    all_energy_preds = []
    all_energy_targets = []
    all_beat_preds = []
    all_beat_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move to device
            audio = batch["audio"].to(device)
            boundary_target = batch["boundary"].cpu().numpy()
            energy_target = batch["energy"].cpu().numpy()
            beat_target = batch["beat"].cpu().numpy()

            # Forward pass
            predictions = model(audio)

            # Convert to numpy
            boundary_pred = predictions["boundary"].cpu().numpy()
            energy_pred = predictions["energy"].cpu().numpy()
            beat_pred = predictions["beat"].cpu().numpy()

            # Accumulate
            for i in range(len(audio)):
                all_boundary_preds.append(boundary_pred[i])
                all_boundary_targets.append(boundary_target[i])
                all_energy_preds.append(energy_pred[i])
                all_energy_targets.append(energy_target[i])
                all_beat_preds.append(beat_pred[i])
                all_beat_targets.append(beat_target[i])

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    print("Computing metrics...")

    # Boundary detection metrics
    boundary_metrics_per_tolerance = []
    tolerances = [0.5, 1.0, 2.0, 3.0]

    for pred, target in zip(all_boundary_preds, all_boundary_targets):
        # Extract boundary times
        pred_times = extract_boundary_times(pred, threshold=0.5)
        target_times = extract_boundary_times(target, threshold=0.5)

        # Calculate F1 at different tolerances
        metrics = boundary_f1_at_tolerances(target_times, pred_times, tolerances)
        boundary_metrics_per_tolerance.append(metrics)

    # Average across samples
    avg_boundary_metrics = {}
    for tol in tolerances:
        precision = np.mean([m[tol]["precision"] for m in boundary_metrics_per_tolerance])
        recall = np.mean([m[tol]["recall"] for m in boundary_metrics_per_tolerance])
        f1 = np.mean([m[tol]["f1"] for m in boundary_metrics_per_tolerance])

        avg_boundary_metrics[f"boundary_f1_@{tol}s"] = f1
        avg_boundary_metrics[f"boundary_precision_@{tol}s"] = precision
        avg_boundary_metrics[f"boundary_recall_@{tol}s"] = recall

    # Energy correlation
    all_energy_preds_concat = np.concatenate(all_energy_preds, axis=0)
    all_energy_targets_concat = np.concatenate(all_energy_targets, axis=0)

    energy_metrics = {}
    if len(all_energy_preds_concat) > 1:
        for band_idx, band_name in enumerate(["bass", "mid", "high"]):
            corr = np.corrcoef(
                all_energy_targets_concat[:, band_idx], all_energy_preds_concat[:, band_idx]
            )[0, 1]
            energy_metrics[f"energy_corr_{band_name}"] = float(corr)

        # Overall correlation
        target_mean = all_energy_targets_concat.mean(axis=1)
        pred_mean = all_energy_preds_concat.mean(axis=1)
        overall_corr = np.corrcoef(target_mean, pred_mean)[0, 1]
        energy_metrics["energy_corr_overall"] = float(overall_corr)

        # MAE
        mae = np.mean(np.abs(all_energy_targets_concat - all_energy_preds_concat))
        energy_metrics["energy_mae"] = float(mae)

    # Beat detection F1 (using boundary F1 logic but for beat detections)
    beat_metrics_per_sample = []
    for pred, target in zip(all_beat_preds, all_beat_targets):
        pred_times = extract_boundary_times(pred, threshold=0.5)
        target_times = extract_boundary_times(target, threshold=0.5)

        # Use 0.07s tolerance for beats (typical beat tracking tolerance)
        metrics = boundary_f1_at_tolerances(target_times, pred_times, [0.07])
        beat_metrics_per_sample.append(metrics[0.07])

    beat_f1 = np.mean([m["f1"] for m in beat_metrics_per_sample])
    beat_precision = np.mean([m["precision"] for m in beat_metrics_per_sample])
    beat_recall = np.mean([m["recall"] for m in beat_metrics_per_sample])

    beat_metrics = {
        "beat_f1": float(beat_f1),
        "beat_precision": float(beat_precision),
        "beat_recall": float(beat_recall),
    }

    # Combine all metrics
    results = {
        "checkpoint": str(checkpoint_path),
        "validation_samples": len(all_boundary_preds),
        "boundary_metrics": avg_boundary_metrics,
        "energy_metrics": energy_metrics,
        "beat_metrics": beat_metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Validation samples: {results['validation_samples']}")
    print("\nBoundary Detection:")
    for key, value in avg_boundary_metrics.items():
        print(f"  {key}: {value:.3f}")
    print("\nEnergy Prediction:")
    for key, value in energy_metrics.items():
        print(f"  {key}: {value:.3f}")
    print("\nBeat Detection:")
    for key, value in beat_metrics.items():
        print(f"  {key}: {value:.3f}")
    print("=" * 60)

    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint file (e.g., experiments/4hr_training/mert95m_4hr_70ep/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations"),
        help="Annotations directory",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path.home() / "music",
        help="Audio directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        annotation_dir=args.annotations,
        audio_dir=args.audio,
        batch_size=args.batch_size,
        device=args.device,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
