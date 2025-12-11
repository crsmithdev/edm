#!/usr/bin/env python3
"""Evaluate trained model with detailed per-sample analysis and visualizations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from edm.evaluation.metrics import boundary_f1_at_tolerances
from edm.models.multitask import create_model
from edm.training.dataset import create_dataloaders


def load_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")

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
    """Extract boundary times from probability predictions."""
    detections = boundary_probs[:, 0] > threshold

    boundaries = []
    for i in range(1, len(detections) - 1):
        if (
            detections[i]
            and boundary_probs[i, 0] > boundary_probs[i - 1, 0]
            and boundary_probs[i, 0] > boundary_probs[i + 1, 0]
        ):
            time_sec = i * 0.1
            boundaries.append(time_sec)

    return boundaries


def compute_sample_loss(pred: dict, target: dict) -> dict:
    """Compute per-sample loss for each task."""
    losses = {}

    # Boundary loss (BCE)
    if "boundary" in pred and "boundary" in target:
        boundary_loss = -np.mean(
            target["boundary"] * np.log(pred["boundary"] + 1e-7)
            + (1 - target["boundary"]) * np.log(1 - pred["boundary"] + 1e-7)
        )
        losses["boundary"] = boundary_loss

    # Energy loss (MSE)
    if "energy" in pred and "energy" in target:
        energy_loss = np.mean((pred["energy"] - target["energy"]) ** 2)
        losses["energy"] = energy_loss

    # Beat loss (BCE)
    if "beat" in pred and "beat" in target:
        beat_loss = -np.mean(
            target["beat"] * np.log(pred["beat"] + 1e-7)
            + (1 - target["beat"]) * np.log(1 - pred["beat"] + 1e-7)
        )
        losses["beat"] = beat_loss

    # Total loss
    losses["total"] = sum(losses.values())

    return losses


def visualize_predictions(
    audio_filename: str,
    predictions: dict,
    targets: dict,
    loss: dict,
    output_dir: Path,
    rank: str,
):
    """Create visualization comparing predictions vs ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"{rank} - {audio_filename}\n"
        f"Total Loss: {loss['total']:.4f} | "
        f"Boundary: {loss.get('boundary', 0):.4f} | "
        f"Energy: {loss.get('energy', 0):.4f} | "
        f"Beat: {loss.get('beat', 0):.4f}",
        fontsize=12,
    )

    time_axis = np.arange(len(predictions["boundary"])) * 0.1  # 10 FPS

    # Boundary Detection
    ax = axes[0]
    ax.plot(time_axis, targets["boundary"], label="Ground Truth", color="green", linewidth=2)
    ax.plot(time_axis, predictions["boundary"], label="Prediction", color="red", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylabel("Boundary Probability")
    ax.set_title("Boundary Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy Prediction
    ax = axes[1]
    band_names = ["Bass", "Mid", "High"]
    colors_target = ["darkgreen", "darkblue", "darkred"]
    colors_pred = ["lightgreen", "lightblue", "lightcoral"]

    for i, (name, c_target, c_pred) in enumerate(zip(band_names, colors_target, colors_pred)):
        ax.plot(
            time_axis,
            targets["energy"][:, i],
            label=f"{name} GT",
            color=c_target,
            linewidth=2,
        )
        ax.plot(
            time_axis,
            predictions["energy"][:, i],
            label=f"{name} Pred",
            color=c_pred,
            alpha=0.7,
            linestyle="--",
        )

    ax.set_ylabel("Energy")
    ax.set_title("Energy Prediction (Bass/Mid/High)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Beat Detection
    ax = axes[2]
    ax.plot(time_axis, targets["beat"], label="Ground Truth", color="green", linewidth=2)
    ax.plot(time_axis, predictions["beat"], label="Prediction", color="red", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Beat Probability")
    ax.set_title("Beat Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    safe_filename = "".join(c if c.isalnum() or c in "._- " else "_" for c in audio_filename)
    output_path = output_dir / f"{rank}_{safe_filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved visualization: {output_path}")


def evaluate_model_detailed(
    checkpoint_path: Path,
    annotation_dir: Path,
    audio_dir: Path | None = None,
    batch_size: int = 4,
    device: str = "cuda",
    output_dir: Path = Path("evaluation_detailed"),
    num_examples: int = 5,
) -> dict:
    """Evaluate model with detailed per-sample analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Get actual filenames from validation dataset (handle Subset wrapper)
    val_dataset = val_loader.dataset
    if hasattr(val_dataset, "dataset"):
        # It's a Subset, get underlying dataset
        base_dataset = val_dataset.dataset
        indices = val_dataset.indices
        val_filenames = [Path(base_dataset.annotations[i][0]).stem for i in indices]
    else:
        # Direct dataset access
        val_filenames = [Path(path).stem for path, _ in val_dataset.annotations]

    # Collect per-sample results
    sample_results = []
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            audio = batch["audio"].to(device)
            boundary_target = batch["boundary"].cpu().numpy()
            energy_target = batch["energy"].cpu().numpy()
            beat_target = batch["beat"].cpu().numpy()

            # Forward pass
            predictions = model(audio)

            boundary_pred = predictions["boundary"].cpu().numpy()
            energy_pred = predictions["energy"].cpu().numpy()
            beat_pred = predictions["beat"].cpu().numpy()

            # Process each sample
            for i in range(len(audio)):
                pred = {
                    "boundary": boundary_pred[i],
                    "energy": energy_pred[i],
                    "beat": beat_pred[i],
                }
                target = {
                    "boundary": boundary_target[i],
                    "energy": energy_target[i],
                    "beat": beat_target[i],
                }

                # Compute loss
                loss = compute_sample_loss(pred, target)

                # Compute boundary F1
                pred_boundaries = extract_boundary_times(pred["boundary"], threshold=0.5)
                target_boundaries = extract_boundary_times(target["boundary"], threshold=0.5)
                boundary_metrics = boundary_f1_at_tolerances(
                    target_boundaries, pred_boundaries, [2.0]
                )

                # Compute energy correlation
                energy_corr = np.corrcoef(
                    target["energy"].mean(axis=1), pred["energy"].mean(axis=1)
                )[0, 1]

                # Get actual filename
                filename = (
                    val_filenames[sample_idx]
                    if sample_idx < len(val_filenames)
                    else f"sample_{sample_idx}"
                )

                sample_results.append(
                    {
                        "filename": filename,
                        "loss": loss,
                        "boundary_f1": boundary_metrics[2.0]["f1"],
                        "energy_corr": energy_corr,
                        "predictions": pred,
                        "targets": target,
                    }
                )

                sample_idx += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    # Sort by total loss
    sample_results.sort(key=lambda x: x["loss"]["total"])

    # Get best and worst examples
    best_samples = sample_results[:num_examples]
    worst_samples = sample_results[-num_examples:]

    print(f"\n{'=' * 60}")
    print(f"TOP {num_examples} BEST PERFORMING TRACKS")
    print(f"{'=' * 60}")
    for i, sample in enumerate(best_samples, 1):
        print(f"{i}. {sample['filename']}")
        print(f"   Total Loss: {sample['loss']['total']:.4f}")
        print(
            f"   Boundary Loss: {sample['loss'].get('boundary', 0):.4f} (F1: {sample['boundary_f1']:.3f})"
        )
        print(
            f"   Energy Loss: {sample['loss'].get('energy', 0):.4f} (Corr: {sample['energy_corr']:.3f})"
        )
        print(f"   Beat Loss: {sample['loss'].get('beat', 0):.4f}")

        # Generate visualization
        visualize_predictions(
            sample["filename"],
            sample["predictions"],
            sample["targets"],
            sample["loss"],
            output_dir,
            f"best_{i:02d}",
        )

    print(f"\n{'=' * 60}")
    print(f"TOP {num_examples} WORST PERFORMING TRACKS")
    print(f"{'=' * 60}")
    for i, sample in enumerate(worst_samples, 1):
        print(f"{i}. {sample['filename']}")
        print(f"   Total Loss: {sample['loss']['total']:.4f}")
        print(
            f"   Boundary Loss: {sample['loss'].get('boundary', 0):.4f} (F1: {sample['boundary_f1']:.3f})"
        )
        print(
            f"   Energy Loss: {sample['loss'].get('energy', 0):.4f} (Corr: {sample['energy_corr']:.3f})"
        )
        print(f"   Beat Loss: {sample['loss'].get('beat', 0):.4f}")

        # Generate visualization
        visualize_predictions(
            sample["filename"],
            sample["predictions"],
            sample["targets"],
            sample["loss"],
            output_dir,
            f"worst_{i:02d}",
        )

    # Save detailed results
    results = {
        "checkpoint": str(checkpoint_path),
        "total_samples": len(sample_results),
        "best_samples": [
            {
                "filename": s["filename"],
                "total_loss": float(s["loss"]["total"]),
                "boundary_loss": float(s["loss"].get("boundary", 0)),
                "energy_loss": float(s["loss"].get("energy", 0)),
                "beat_loss": float(s["loss"].get("beat", 0)),
                "boundary_f1": float(s["boundary_f1"]),
                "energy_corr": float(s["energy_corr"]) if not np.isnan(s["energy_corr"]) else None,
            }
            for s in best_samples
        ],
        "worst_samples": [
            {
                "filename": s["filename"],
                "total_loss": float(s["loss"]["total"]),
                "boundary_loss": float(s["loss"].get("boundary", 0)),
                "energy_loss": float(s["loss"].get("energy", 0)),
                "beat_loss": float(s["loss"].get("beat", 0)),
                "boundary_f1": float(s["boundary_f1"]),
                "energy_corr": float(s["energy_corr"]) if not np.isnan(s["energy_corr"]) else None,
            }
            for s in worst_samples
        ],
    }

    results_file = output_dir / "detailed_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Detailed results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Detailed model evaluation with visualizations")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint file",
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
        help="Batch size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_detailed"),
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of best/worst examples to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    args = parser.parse_args()

    evaluate_model_detailed(
        checkpoint_path=args.checkpoint,
        annotation_dir=args.annotations,
        audio_dir=args.audio,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
