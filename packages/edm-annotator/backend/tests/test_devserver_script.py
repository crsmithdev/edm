"""Unit tests for dev server startup script.

Fast tests that validate the run-dev.sh script without starting servers.
"""

import os
from pathlib import Path

import pytest


class TestDevServerScript:
    """Test dev server script exists and is configured correctly."""

    @pytest.fixture
    def script_path(self):
        """Get path to run-dev.sh."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / "run-dev.sh"

    def test_script_exists(self, script_path):
        """run-dev.sh script exists in expected location."""
        assert script_path.exists(), f"Dev server script not found: {script_path}"

    def test_script_is_executable(self, script_path):
        """run-dev.sh has executable permissions."""
        assert os.access(script_path, os.X_OK), f"Script is not executable: {script_path}"

    def test_script_has_shebang(self, script_path):
        """run-dev.sh starts with valid shebang."""
        with open(script_path) as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!"), "Script missing shebang"
        assert "bash" in first_line, "Script should use bash"

    def test_script_uses_correct_package_manager(self, script_path):
        """run-dev.sh uses npm (not pnpm) for frontend."""
        content = script_path.read_text()
        assert "npm run dev" in content, "Script should use 'npm run dev' for frontend"
        assert "npm install" in content, "Script should reference 'npm install'"
        # Should not reference pnpm
        assert "pnpm run dev" not in content, "Script should not use pnpm"

    def test_script_starts_both_servers(self, script_path):
        """run-dev.sh references both backend and frontend."""
        content = script_path.read_text()
        assert "edm-annotator" in content, "Script should start backend (edm-annotator)"
        assert "npm run dev" in content, "Script should start frontend (npm run dev)"

    def test_script_has_cleanup_handler(self, script_path):
        """run-dev.sh has proper cleanup/shutdown handler."""
        content = script_path.read_text()
        assert "cleanup" in content, "Script should have cleanup function"
        assert "SIGINT" in content or "trap" in content, "Script should trap signals"

    def test_script_has_prerequisite_checks(self, script_path):
        """run-dev.sh checks prerequisites before starting."""
        content = script_path.read_text()
        assert "node_modules" in content, "Script should check for frontend dependencies"
        assert (
            "edm-annotator --help" in content or "uv run" in content
        ), "Script should check backend installation"
