"""Shared pytest fixtures for EDM tests.

This module provides common fixtures used across unit, integration, and
performance tests. Import fixtures by name in test files - pytest
discovers them automatically.
"""

from pathlib import Path

import pytest

# ==============================================================================
# Path Fixtures
# ==============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory containing audio files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def estimations_dir() -> Path:
    """Path to reference estimations directory containing JAMS files."""
    return Path(__file__).parent / "estimations"


# ==============================================================================
# Pytest Configuration
# ==============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "performance: marks performance benchmark tests")
    config.addinivalue_line("markers", "manual: marks tests requiring manual verification")
