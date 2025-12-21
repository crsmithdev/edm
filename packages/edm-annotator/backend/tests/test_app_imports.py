"""Test that app imports and critical dependencies are available.

This test validates that the application can be imported successfully and
that critical dependencies (like edm-lib) are properly installed. This
prevents runtime errors when the server starts or when endpoints are accessed.
"""

import pytest


class TestAppImports:
    """Test application imports and dependency availability."""

    def test_app_module_imports(self):
        """Application module can be imported without errors."""
        from edm_annotator.app import create_app

        assert create_app is not None

    def test_app_factory_creates_app(self):
        """Application factory creates Flask app instance."""
        from edm_annotator.app import create_app

        app = create_app("development")
        assert app is not None
        assert hasattr(app, "config")

    def test_edm_lib_dependency_available(self):
        """EDM library dependency is installed and importable."""
        # This was the cause of the track loading failure - missing edm.data.metadata
        from edm.data.metadata import AnnotationMetadata
        from edm.data.schema import Annotation, AudioMetadata, StructureSection

        assert AnnotationMetadata is not None
        assert Annotation is not None
        assert AudioMetadata is not None
        assert StructureSection is not None

    def test_services_import(self):
        """Service modules can be imported successfully."""
        from edm_annotator.services import (
            AnnotationService,
            AudioService,
            WaveformService,
        )

        assert AnnotationService is not None
        assert AudioService is not None
        assert WaveformService is not None

    def test_api_routes_import(self):
        """API route modules can be imported successfully."""
        from edm_annotator.api import register_routes

        assert register_routes is not None


class TestTrackLoadingEndpoint:
    """Test that the track loading endpoint works correctly."""

    @pytest.fixture
    def client(self):
        """Create test client for Flask app."""
        from edm_annotator.app import create_app

        app = create_app("testing")
        with app.test_client() as client:
            yield client

    def test_tracks_endpoint_exists(self, client):
        """GET /api/tracks endpoint exists and responds."""
        response = client.get("/api/tracks")
        # Should return 200 OK with a list (even if empty)
        assert response.status_code == 200
        assert isinstance(response.json, list)

    def test_tracks_endpoint_no_500_error(self, client):
        """GET /api/tracks does not return 500 Internal Server Error.

        Regression test for issue where missing edm-lib dependency caused
        500 errors when loading tracks due to import failures.
        """
        response = client.get("/api/tracks")
        assert response.status_code != 500, (
            "Track loading endpoint returned 500 error. "
            "This may indicate missing dependencies (check edm-lib is installed) "
            "or import errors in the service layer."
        )

    def test_track_load_endpoint_exists(self, client):
        """POST /api/tracks/load endpoint exists and responds."""
        # Try to load a non-existent track - should return 404, not 500
        response = client.post(
            "/api/tracks/load",
            json={"filename": "nonexistent-track.mp3"},
            content_type="application/json",
        )
        # Should not crash with 500 - either 404 (not found) or 400 (bad request)
        assert response.status_code != 500, (
            "Track load endpoint returned 500 error. "
            "This indicates a server-side crash rather than a client error."
        )
        assert response.status_code in [400, 404]
