"""Cleanlab integration for detecting label errors in training data."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from torch.utils.data import DataLoader

from edm.models.multitask import MultiTaskModel
from edm.training.dataset import EDMDataset, collate_fn


def cross_val_predictions(
    model: MultiTaskModel,
    dataset: EDMDataset,
    n_splits: int = 5,
    batch_size: int = 4,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Generate cross-validation predictions for cleanlab analysis.

    Args:
        model: Trained model
        dataset: Full dataset
        n_splits: Number of CV folds
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        Dict with:
            - label_probs: [num_samples, num_frames, num_classes]
            - label_true: [num_samples, num_frames]
            - boundary_probs: [num_samples, num_frames]
            - boundary_true: [num_samples, num_frames]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create k-fold splits
    from torch.utils.data import Subset

    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    fold_size = dataset_size // n_splits
    folds = [indices[i * fold_size : (i + 1) * fold_size].tolist() for i in range(n_splits)]

    # Storage for predictions
    all_label_probs = []
    all_label_true = []
    all_boundary_probs = []
    all_boundary_true = []

    with torch.no_grad():
        for fold_idx in range(n_splits):
            # Get validation indices for this fold
            val_indices = folds[fold_idx]
            val_dataset = Subset(dataset, val_indices)

            # Create dataloader
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

            # Run inference
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

                # Forward pass
                predictions = model(batch["audio"])

                # Store label predictions (convert logits to probs)
                if "label" in predictions:
                    label_logits = predictions["label"]  # [batch, frames, num_classes]
                    label_probs = torch.softmax(label_logits, dim=-1)
                    all_label_probs.append(label_probs.cpu().numpy())
                    all_label_true.append(batch["label"].cpu().numpy())

                # Store boundary predictions
                if "boundary" in predictions:
                    boundary_probs = predictions["boundary"]  # [batch, frames, 1]
                    all_boundary_probs.append(boundary_probs.squeeze(-1).cpu().numpy())
                    all_boundary_true.append(batch["boundary"].squeeze(-1).cpu().numpy())

    # Concatenate all predictions
    result = {}

    if all_label_probs:
        result["label_probs"] = np.concatenate(all_label_probs, axis=0)
        result["label_true"] = np.concatenate(all_label_true, axis=0)

    if all_boundary_probs:
        result["boundary_probs"] = np.concatenate(all_boundary_probs, axis=0)
        result["boundary_true"] = np.concatenate(all_boundary_true, axis=0)

    return result


def detect_label_errors(
    predictions: dict[str, np.ndarray],
    threshold: float = 0.3,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Detect potential label errors using cleanlab.

    Args:
        predictions: Dict from cross_val_predictions
        threshold: Confidence threshold for flagging errors
        min_confidence: Minimum confidence to consider (filters uncertain predictions)

    Returns:
        Dict with error analysis:
            - label_issues: Boolean mask [num_samples, num_frames]
            - label_quality: Quality scores [num_samples, num_frames]
            - boundary_issues: Boolean mask [num_samples, num_frames]
            - summary: Dict with error statistics
    """
    result = {}

    # Analyze label errors
    if "label_probs" in predictions:
        label_probs = predictions["label_probs"]  # [samples, frames, classes]
        label_true = predictions["label_true"]  # [samples, frames]

        # Reshape for cleanlab: [samples * frames, classes]
        num_samples, num_frames, num_classes = label_probs.shape
        probs_flat = label_probs.reshape(-1, num_classes)
        labels_flat = label_true.reshape(-1)

        # Get quality scores
        quality_scores = get_label_quality_scores(
            labels=labels_flat,
            pred_probs=probs_flat,
        )

        # Find issues
        issues_mask = find_label_issues(
            labels=labels_flat,
            pred_probs=probs_flat,
            return_indices_ranked_by="self_confidence",
        )

        # Reshape back
        result["label_quality"] = quality_scores.reshape(num_samples, num_frames)
        result["label_issues"] = issues_mask.reshape(num_samples, num_frames)

        # Summary statistics
        total_frames = num_samples * num_frames
        num_issues = issues_mask.sum()
        result["label_summary"] = {
            "total_frames": total_frames,
            "num_issues": int(num_issues),
            "error_rate": float(num_issues / total_frames),
            "avg_quality": float(quality_scores.mean()),
        }

    # Analyze boundary errors
    if "boundary_probs" in predictions:
        boundary_probs = predictions["boundary_probs"]  # [samples, frames]
        boundary_true = predictions["boundary_true"]  # [samples, frames]

        # Simple threshold-based error detection
        # Flag frames where prediction strongly disagrees with label
        disagreement = np.abs(boundary_probs - boundary_true)
        high_confidence = np.maximum(boundary_probs, boundary_true) > min_confidence

        boundary_issues = (disagreement > threshold) & high_confidence

        result["boundary_issues"] = boundary_issues
        result["boundary_disagreement"] = disagreement

        # Summary
        total_frames = boundary_issues.size
        num_issues = boundary_issues.sum()
        result["boundary_summary"] = {
            "total_frames": total_frames,
            "num_issues": int(num_issues),
            "error_rate": float(num_issues / total_frames),
            "avg_disagreement": float(disagreement.mean()),
        }

    return result


def generate_error_report(
    dataset: EDMDataset,
    error_analysis: dict[str, Any],
    output_path: Path,
    top_k: int = 50,
) -> None:
    """Generate human-readable error report.

    Args:
        dataset: Dataset used for predictions
        error_analysis: Results from detect_label_errors
        output_path: Path to save report
        top_k: Number of top errors to include in report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CLEANLAB ERROR DETECTION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Label error summary
    if "label_summary" in error_analysis:
        summary = error_analysis["label_summary"]
        lines.append("LABEL ERRORS:")
        lines.append(f"  Total frames: {summary['total_frames']}")
        lines.append(f"  Flagged issues: {summary['num_issues']}")
        lines.append(f"  Error rate: {summary['error_rate']:.2%}")
        lines.append(f"  Avg quality: {summary['avg_quality']:.4f}")
        lines.append("")

        # Find worst examples (per-sample error rate)
        label_issues = error_analysis["label_issues"]
        sample_error_rates = label_issues.mean(axis=1)  # Average per sample
        worst_samples = np.argsort(sample_error_rates)[::-1][:top_k]

        lines.append(f"TOP {top_k} SAMPLES WITH LABEL ERRORS:")
        lines.append("")

        for rank, sample_idx in enumerate(worst_samples):
            error_rate = sample_error_rates[sample_idx]
            if error_rate == 0:
                break

            # Get sample info
            yaml_path, annotation = dataset.annotations[sample_idx]
            track_name = yaml_path.stem

            lines.append(f"{rank + 1}. {track_name}")
            lines.append(f"   Error rate: {error_rate:.2%}")
            lines.append(f"   File: {yaml_path}")
            lines.append("")

    # Boundary error summary
    if "boundary_summary" in error_analysis:
        summary = error_analysis["boundary_summary"]
        lines.append("-" * 80)
        lines.append("BOUNDARY ERRORS:")
        lines.append(f"  Total frames: {summary['total_frames']}")
        lines.append(f"  Flagged issues: {summary['num_issues']}")
        lines.append(f"  Error rate: {summary['error_rate']:.2%}")
        lines.append(f"  Avg disagreement: {summary['avg_disagreement']:.4f}")
        lines.append("")

        # Find samples with most boundary issues
        boundary_issues = error_analysis["boundary_issues"]
        sample_boundary_errors = boundary_issues.sum(axis=1)
        worst_samples = np.argsort(sample_boundary_errors)[::-1][:top_k]

        lines.append(f"TOP {top_k} SAMPLES WITH BOUNDARY ERRORS:")
        lines.append("")

        for rank, sample_idx in enumerate(worst_samples):
            num_errors = sample_boundary_errors[sample_idx]
            if num_errors == 0:
                break

            yaml_path, annotation = dataset.annotations[sample_idx]
            track_name = yaml_path.stem

            lines.append(f"{rank + 1}. {track_name}")
            lines.append(f"   Boundary errors: {int(num_errors)}")
            lines.append(f"   File: {yaml_path}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Error report saved to: {output_path}")


def save_error_metadata(
    dataset: EDMDataset,
    error_analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save per-sample error metadata for review.

    Creates YAML files with error flags and quality scores for each sample.

    Args:
        dataset: Dataset used for predictions
        error_analysis: Results from detect_label_errors
        output_dir: Directory to save metadata files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    for sample_idx in range(len(dataset)):
        yaml_path, annotation = dataset.annotations[sample_idx]

        # Collect error info for this sample
        error_info: dict[str, str | float | int] = {
            "source_annotation": str(yaml_path),
            "track": yaml_path.stem,
        }

        # Label errors
        if "label_issues" in error_analysis:
            label_issues = error_analysis["label_issues"][sample_idx]
            label_quality = error_analysis["label_quality"][sample_idx]

            error_info["label_error_rate"] = float(label_issues.mean())
            error_info["avg_label_quality"] = float(label_quality.mean())
            error_info["min_label_quality"] = float(label_quality.min())

        # Boundary errors
        if "boundary_issues" in error_analysis:
            boundary_issues = error_analysis["boundary_issues"][sample_idx]
            boundary_disagreement = error_analysis["boundary_disagreement"][sample_idx]

            error_info["boundary_errors"] = int(boundary_issues.sum())
            error_info["avg_boundary_disagreement"] = float(boundary_disagreement.mean())

        # Save to file
        output_file = output_dir / f"{yaml_path.stem}_errors.yaml"
        with open(output_file, "w") as f:
            yaml.dump(error_info, f, default_flow_style=False, sort_keys=False)

    print(f"Error metadata saved to: {output_dir}")
