"""Accuracy evaluation framework for EDM analysis algorithms.

This module provides tools for systematically testing and validating
the accuracy of analysis algorithms (BPM detection, drop detection, etc.)
against reference data.

This is internal developer tooling, not part of the public API.
"""

from edm.evaluation.evaluators.bpm import evaluate_bpm

__all__ = ["evaluate_bpm"]
