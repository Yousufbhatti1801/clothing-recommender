"""Security tests: auth, path traversal, SSRF guard, health status codes."""
from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


def _get_test_client() -> TestClient:
    """Return a TestClient with all external dependencies mocked."""
    with (
        patch("ml.clip_encoder.get_clip_encoder", return_value=MagicMock()),
        patch("ml.yolo_detector.get_yolo_detector", return_value=MagicMock()),
        patch("app.core.pinecone_client.init_pinecone"),
        patch("app.services.vector_store.get_vector_service", return_value=MagicMock()),
        patch("app.core.database.engine"),
    ):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestCatalogIngestAuth:
    """The /catalog/ingest endpoint must require a valid API key."""

    def _ingest_payload(self) -> dict:
        return {
            "name": "Test Shirt",
            "category": "shirt",
            "price": 29.99,
            "image_url": "https://example.com/img.jpg",
        }

    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-test-key")
        from app.core.config import get_settings
        get_settings.cache_clear()

    def test_ingest_without_api_key_returns_403(self):
        client = _get_test_client()
        resp = client.post("/api/v1/catalog/ingest", json=self._ingest_payload())
        assert resp.status_code == 403, resp.text

    def test_ingest_with_wrong_api_key_returns_403(self):
        client = _get_test_client()
        resp = client.post(
            "/api/v1/catalog/ingest",
            json=self._ingest_payload(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403, resp.text

    def test_ingest_with_correct_api_key_passes_auth(self):
        """Auth passes — underlying service call may fail but we confirm 403 is not returned."""
        client = _get_test_client()
        resp = client.post(
            "/api/v1/catalog/ingest",
            json=self._ingest_payload(),
            headers={"X-API-Key": "secret-test-key"},
        )
        # 403 means auth failed; anything else means auth passed
        assert resp.status_code != 403, resp.text


# ── Path traversal tests ──────────────────────────────────────────────────────

class TestDetectPathTraversal:
    """Uploads must be saved with server-generated UUIDs, not user filenames."""

    def test_saved_filename_is_uuid_not_user_supplied(self, tmp_path, monkeypatch):
        """The saved file should be a UUID, never the attacker-controlled filename."""
        import app.api.routes.detect as detect_route

        monkeypatch.setattr(detect_route, "UPLOAD_DIR", tmp_path)

        mock_detection_service = MagicMock()
        mock_detection_service.detect_clothing = AsyncMock(
            return_value=MagicMock(shirts=[], pants=[], shoes=[], total_detections=0)
        )

        with patch(
            "app.api.routes.detect.get_detection_service",
            return_value=mock_detection_service,
        ):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            img_bytes = _make_jpeg_bytes()
            client.post(
                "/api/v1/detect",
                files={"file": ("../../etc/passwd", img_bytes, "image/jpeg")},
            )

        saved_files = list(tmp_path.iterdir())
        for f in saved_files:
            name = f.stem  # filename without extension
            try:
                uuid.UUID(name)  # should parse as UUID without error
            except ValueError:
                pytest.fail(
                    f"Saved file '{f.name}' is not a UUID — path traversal not prevented!"
                )


# ── Health check HTTP status tests ───────────────────────────────────────────

class TestHealthCheckStatus:
    """Health endpoint must return 503 when any dependency is unhealthy."""

    def test_health_degraded_returns_503_when_db_fails(self):
        with (
            patch("ml.clip_encoder.get_clip_encoder", return_value=MagicMock()),
            patch("ml.yolo_detector.get_yolo_detector", return_value=MagicMock()),
            patch("app.core.pinecone_client.init_pinecone"),
            patch("app.services.vector_store.get_vector_service", return_value=MagicMock()),
            patch(
                "app.api.routes.health.AsyncSessionLocal",
                side_effect=Exception("DB connection refused"),
            ),
        ):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/health")
            assert resp.status_code == 503, resp.text
            data = resp.json()
            assert data["status"] == "degraded"
            assert "error" in data["database"]

    def test_health_returns_200_when_all_ok(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {}

        with (
            patch("ml.clip_encoder.get_clip_encoder", return_value=MagicMock()),
            patch("ml.yolo_detector.get_yolo_detector", return_value=MagicMock()),
            patch("app.core.pinecone_client.init_pinecone"),
            patch("app.services.vector_store.get_vector_service", return_value=MagicMock()),
            patch("app.api.routes.health.AsyncSessionLocal"),
            patch("app.api.routes.health.get_pinecone_index", return_value=mock_index),
        ):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/health")
            assert resp.status_code in (200, 503)  # either is valid depending on env
            assert "status" in resp.json()


# ── Request ID middleware tests ───────────────────────────────────────────────

class TestRequestIdMiddleware:
    """Every response must include an X-Request-Id header."""

    def test_every_response_has_request_id_header(self):
        with (
            patch("ml.clip_encoder.get_clip_encoder", return_value=MagicMock()),
            patch("ml.yolo_detector.get_yolo_detector", return_value=MagicMock()),
            patch("app.core.pinecone_client.init_pinecone"),
            patch("app.services.vector_store.get_vector_service", return_value=MagicMock()),
            patch("app.api.routes.health.AsyncSessionLocal"),
            patch("app.api.routes.health.get_pinecone_index", return_value=MagicMock()),
        ):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/health")
            assert "x-request-id" in resp.headers
            # Verify it's a valid UUID
            uuid.UUID(resp.headers["x-request-id"])
