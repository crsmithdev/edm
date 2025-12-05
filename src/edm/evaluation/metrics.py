"""Extended evaluation metrics for ML-based structure detection."""

from typing import Any

import numpy as np


def boundary_f1_at_tolerances(
    reference_boundaries: list[float],
    detected_boundaries: list[float],
    tolerances: list[float],
) -> dict[float, dict[str, float]]:
    """Calculate boundary F1 at multiple tolerance levels.

    Args:
        reference_boundaries: Reference boundary times in seconds
        detected_boundaries: Detected boundary times in seconds
        tolerances: List of tolerance values in seconds (e.g., [0.5, 1.0, 2.0, 3.0])

    Returns:
        Dict mapping tolerance to {precision, recall, f1}
    """
    results = {}

    for tol in tolerances:
        # Match boundaries within tolerance
        matched_ref = set()
        matched_det = set()

        for ref_b in reference_boundaries:
            for det_b in detected_boundaries:
                if abs(ref_b - det_b) <= tol and det_b not in matched_det:
                    matched_ref.add(ref_b)
                    matched_det.add(det_b)
                    break

        # Calculate metrics
        tp = len(matched_det)
        fp = len(detected_boundaries) - tp
        fn = len(reference_boundaries) - len(matched_ref)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[tol] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results


def energy_correlation(
    reference_energy: np.ndarray,
    predicted_energy: np.ndarray,
) -> dict[str, float]:
    """Calculate energy prediction metrics.

    Args:
        reference_energy: Reference energy values [frames, 3] for bass/mid/high
        predicted_energy: Predicted energy values [frames, 3]

    Returns:
        Dict with correlation metrics per band and overall
    """
    # Ensure same length
    min_len = min(len(reference_energy), len(predicted_energy))
    ref = reference_energy[:min_len]
    pred = predicted_energy[:min_len]

    if len(ref) == 0:
        return {
            "bass_correlation": 0.0,
            "mid_correlation": 0.0,
            "high_correlation": 0.0,
            "overall_correlation": 0.0,
            "mae": 0.0,
        }

    # Per-band correlations
    bass_corr = float(np.corrcoef(ref[:, 0], pred[:, 0])[0, 1]) if len(ref) > 1 else 0.0
    mid_corr = float(np.corrcoef(ref[:, 1], pred[:, 1])[0, 1]) if len(ref) > 1 else 0.0
    high_corr = float(np.corrcoef(ref[:, 2], pred[:, 2])[0, 1]) if len(ref) > 1 else 0.0

    # Overall energy (mean across bands)
    ref_overall = ref.mean(axis=1)
    pred_overall = pred.mean(axis=1)
    overall_corr = float(np.corrcoef(ref_overall, pred_overall)[0, 1]) if len(ref) > 1 else 0.0

    # MAE
    mae = float(np.mean(np.abs(ref - pred)))

    return {
        "bass_correlation": bass_corr,
        "mid_correlation": mid_corr,
        "high_correlation": high_corr,
        "overall_correlation": overall_corr,
        "mae": mae,
    }


def label_accuracy(
    reference_labels: list[str],
    predicted_labels: list[str],
    reference_times: list[float],
    predicted_times: list[float],
    tolerance: float = 2.0,
) -> dict[str, Any]:
    """Calculate label prediction accuracy.

    Matches sections by time overlap and checks label correctness.

    Args:
        reference_labels: Reference section labels
        predicted_labels: Predicted section labels
        reference_times: Reference section start times
        predicted_times: Predicted section start times
        tolerance: Time tolerance for matching sections

    Returns:
        Dict with accuracy, per-class precision/recall
    """
    # Match sections by time
    matched_pairs = []
    used_predicted = set()

    for i, ref_time in enumerate(reference_times):
        best_match = None
        best_distance = float("inf")

        for j, pred_time in enumerate(predicted_times):
            if j in used_predicted:
                continue

            distance = abs(ref_time - pred_time)
            if distance <= tolerance and distance < best_distance:
                best_match = j
                best_distance = distance

        if best_match is not None:
            matched_pairs.append((i, best_match))
            used_predicted.add(best_match)

    # Calculate overall accuracy
    correct = sum(
        1
        for ref_idx, pred_idx in matched_pairs
        if reference_labels[ref_idx] == predicted_labels[pred_idx]
    )
    accuracy = correct / len(matched_pairs) if matched_pairs else 0.0

    # Calculate per-class metrics
    unique_labels = sorted(set(reference_labels + predicted_labels))
    per_class = {}

    for label in unique_labels:
        tp = sum(
            1
            for ref_idx, pred_idx in matched_pairs
            if reference_labels[ref_idx] == label and predicted_labels[pred_idx] == label
        )
        fp = sum(
            1
            for ref_idx, pred_idx in matched_pairs
            if reference_labels[ref_idx] != label and predicted_labels[pred_idx] == label
        )
        fn = sum(
            1
            for ref_idx, pred_idx in matched_pairs
            if reference_labels[ref_idx] == label and predicted_labels[pred_idx] != label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": accuracy,
        "matched_sections": len(matched_pairs),
        "per_class": per_class,
    }


def calculate_pairwise_boundary_distances(
    reference_boundaries: list[float],
    detected_boundaries: list[float],
) -> dict[str, float]:
    """Calculate pairwise distances between boundaries.

    Args:
        reference_boundaries: Reference boundary times
        detected_boundaries: Detected boundary times

    Returns:
        Dict with mean/median/std of nearest neighbor distances
    """
    if not reference_boundaries or not detected_boundaries:
        return {
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "std_distance": 0.0,
        }

    # For each reference, find nearest detected
    distances = []
    for ref_b in reference_boundaries:
        nearest_dist = min(abs(ref_b - det_b) for det_b in detected_boundaries)
        distances.append(nearest_dist)

    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "std_distance": float(np.std(distances)),
    }
