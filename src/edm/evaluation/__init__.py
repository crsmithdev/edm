"""Accuracy evaluation framework for EDM analysis algorithms.

This module provides tools for systematically testing and validating
the accuracy of analysis algorithms (BPM detection, structure detection, etc.)
against reference data.

This is internal developer tooling, not part of the public API.
"""

from edm.evaluation.evaluators.bpm import evaluate_bpm
from edm.evaluation.evaluators.structure import evaluate_structure

__all__ = ["evaluate_bpm", "evaluate_structure"]
