"""Integration test for dev server startup.

Validates that run-dev.sh successfully starts both backend and frontend
servers and that they are responsive.
"""

import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestDevServerStartup:
    """Test dev server startup script."""

    @pytest.fixture
    def devserver_process(self):
        """Start dev server and yield process, then clean up."""
        # Get path to run-dev.sh
        repo_root = Path(__file__).parent.parent.parent.parent
        script_path = repo_root / "run-dev.sh"

        if not script_path.exists():
            pytest.skip(f"Dev server script not found: {script_path}")

        # Check prerequisites
        if not (repo_root / "frontend" / "node_modules").exists():
            pytest.skip("Frontend dependencies not installed")

        # Start the dev server
        process = subprocess.Popen(
            [str(script_path)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )

        yield process

        # Cleanup: terminate the process and its children
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def test_backend_starts(self, devserver_process):
        """Backend server starts and responds to API requests."""
        backend_url = "http://localhost:5001/api/tracks"
        max_wait = 15  # seconds

        # Wait for backend to be ready
        for _ in range(max_wait):
            try:
                response = requests.get(backend_url, timeout=2)
                if response.status_code == 200:
                    # Backend is up and responding
                    assert isinstance(response.json(), list)
                    return
            except requests.exceptions.RequestException:
                # Server not ready yet
                time.sleep(1)

        pytest.fail(f"Backend did not start within {max_wait} seconds")

    def test_frontend_starts(self, devserver_process):
        """Frontend server starts and port is listening."""
        frontend_url = "http://localhost:5174"
        max_wait = 20  # seconds (Vite can take a bit longer)

        # Wait for frontend to be ready
        for _ in range(max_wait):
            try:
                response = requests.get(frontend_url, timeout=2)
                if response.status_code in [200, 304]:
                    # Frontend is up
                    return
            except requests.exceptions.RequestException:
                # Server not ready yet
                time.sleep(1)

        pytest.fail(f"Frontend did not start within {max_wait} seconds")

    def test_both_servers_running(self, devserver_process):
        """Both backend and frontend are running simultaneously."""
        backend_url = "http://localhost:5001/api/tracks"
        frontend_url = "http://localhost:5174"
        max_wait = 20

        backend_ready = False
        frontend_ready = False

        # Wait for both servers
        for _ in range(max_wait):
            # Check backend
            if not backend_ready:
                try:
                    response = requests.get(backend_url, timeout=2)
                    if response.status_code == 200:
                        backend_ready = True
                except requests.exceptions.RequestException:
                    pass

            # Check frontend
            if not frontend_ready:
                try:
                    response = requests.get(frontend_url, timeout=2)
                    if response.status_code in [200, 304]:
                        frontend_ready = True
                except requests.exceptions.RequestException:
                    pass

            # Both ready
            if backend_ready and frontend_ready:
                # Verify they're still both up
                assert requests.get(backend_url, timeout=2).status_code == 200
                assert requests.get(frontend_url, timeout=2).status_code in [200, 304]
                return

            time.sleep(1)

        failures = []
        if not backend_ready:
            failures.append("Backend")
        if not frontend_ready:
            failures.append("Frontend")

        pytest.fail(f"{', '.join(failures)} did not start within {max_wait} seconds")

    def test_devserver_process_alive(self, devserver_process):
        """Dev server process stays alive and doesn't crash."""
        # Give servers time to start
        time.sleep(5)

        # Check process is still running
        poll_result = devserver_process.poll()
        if poll_result is not None:
            # Process has terminated
            stdout, stderr = devserver_process.communicate()
            pytest.fail(
                f"Dev server process terminated with code {poll_result}\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )

        # Process is alive
        assert devserver_process.poll() is None
