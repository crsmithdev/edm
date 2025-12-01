"""Performance profiling infrastructure for EDM analysis."""

from edm.profiling.base import ProfilerBase, ProfileResult, profiling_context
from edm.profiling.baseline import (
    Baseline,
    BaselineStore,
    ComparisonResult,
    compare_baseline,
    create_baseline,
    load_and_compare,
    save_baseline,
)
from edm.profiling.cpu import CPUProfiler
from edm.profiling.decorator import profile
from edm.profiling.memory import MemoryProfiler, MemorySnapshot, compare_memory
from edm.profiling.reports import ProfileReport, format_comparison_text, format_profile_text

__all__ = [
    "Baseline",
    "BaselineStore",
    "CPUProfiler",
    "ComparisonResult",
    "MemoryProfiler",
    "MemorySnapshot",
    "ProfileReport",
    "ProfileResult",
    "ProfilerBase",
    "compare_baseline",
    "compare_memory",
    "create_baseline",
    "format_comparison_text",
    "format_profile_text",
    "load_and_compare",
    "profile",
    "profiling_context",
    "save_baseline",
]
