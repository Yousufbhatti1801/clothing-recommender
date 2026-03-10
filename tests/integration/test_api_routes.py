"""Integration tests for FastAPI routes — app-level, mocked ML + Pinecone."""
from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image
from httpx import ASGITransport, AsyncClient

from app.models.schemas import (
    BoundingBox,
    DetectedGarment,
    GarmentCategory,
    GarmentRecommendations,
    PipelineCategoryResult,
    PipelineMatch,
    PipelineRecommendationResponse,
    ProductResponse,
    RecommendationResponse,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (640, 640), color=(128, 128, 128)).save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test client fixture (patches heavy singletons so no weights are loaded)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def client():
    """
    httpx AsyncClient backed by the real FastAPI app, but with all heavy
    external dependencies mocked:
      - YOLOv8  → MagicMock detector
      - CLIP    → MagicMock encoder
      - Pinecone → MagicMock client + index (prevents real HTTP calls)
      - PostgreSQL engine → AsyncMock (no real DB)
      - RecommendationService → overridden via FastAPI dependency_overrides
      - _pipeline singleton   → replaced with a stub after import
    """
    # ── 1. Build mock YOLO ────────────────────────────────────────────────
    mock_yolo = MagicMock()
    _yolo_detections = [
        DetectedGarment(
            category=GarmentCategory.SHIRT,
            bounding_box=BoundingBox(x_min=50, y_min=50, x_max=400, y_max=350, confidence=0.92),
        ),
        DetectedGarment(
            category=GarmentCategory.PANTS,
            bounding_box=BoundingBox(x_min=80, y_min=360, x_max=400, y_max=600, confidence=0.85),
        ),
    ]
    mock_yolo.detect = MagicMock(return_value=_yolo_detections)
    mock_yolo.detect_targets = MagicMock(return_value=_yolo_detections)

    async def _detect_async(img):
        return mock_yolo.detect(img)

    async def _detect_targets_async(img, cats=None):
        return mock_yolo.detect_targets(img)

    mock_yolo.detect_async = _detect_async
    mock_yolo.detect_targets_async = _detect_targets_async

    # ── 2. Build mock CLIP ────────────────────────────────────────────────
    mock_clip = MagicMock()

    def _encode(images):
        return np.random.rand(len(images), 512).astype(np.float32)

    async def _encode_async(images):
        return _encode(images)

    mock_clip.encode = MagicMock(side_effect=_encode)
    mock_clip.encode_async = AsyncMock(side_effect=_encode_async)

    # ── 3. Build mock Pinecone class + index ──────────────────────────────
    mock_index = MagicMock()
    mock_index.query = MagicMock(return_value={
        "matches": [
            {"id": str(uuid.uuid4()), "score": 0.91, "metadata": {}},
            {"id": str(uuid.uuid4()), "score": 0.85, "metadata": {}},
        ]
    })
    mock_index.describe_index_stats = MagicMock(return_value={"dimension": 512})
    mock_index.upsert = MagicMock()

    # PineconeVectorService.__init__ calls Pinecone(...).list_indexes() and .Index(...)
    mock_pc_instance = MagicMock()
    mock_pc_instance.list_indexes.return_value = [{"name": "clothing-embeddings"}]
    mock_pc_instance.Index.return_value = mock_index
    mock_pc_class = MagicMock(return_value=mock_pc_instance)

    # ── 4. Build recommendation service stub ──────────────────────────────
    rec_product = ProductResponse(
        id=uuid.uuid4(), name="Mock Shirt", brand="MockBrand",
        category=GarmentCategory.SHIRT, price=49.99, currency="USD",
    )
    rec_response = RecommendationResponse(
        results=[GarmentRecommendations(category=GarmentCategory.SHIRT, items=[rec_product])],
        detected_items=[GarmentCategory.SHIRT],
        total_matches=1,
    )
    recommendation_service = MagicMock()
    recommendation_service.recommend = AsyncMock(return_value=rec_response)

    # ── 5. Build pipeline stub ────────────────────────────────────────────
    pipeline_response = PipelineRecommendationResponse(
        shirts=[
            PipelineCategoryResult(
                category=GarmentCategory.SHIRT,
                detection_confidence=0.95,
                matches=[PipelineMatch(product_id="prod-1", score=0.9, metadata={"brand": "Mock"})],
            )
        ],
        pants=[], shoes=[],
        total_detections=1, total_matches=1,
    )
    pipeline_stub = MagicMock()
    pipeline_stub.run = AsyncMock(return_value=pipeline_response)

    # ── 6. Apply patches and boot the app ─────────────────────────────────
    # Patch order matters: Pinecone + YOLO + CLIP must be active BEFORE
    # app.api.routes.pipeline is imported (it instantiates _pipeline at
    # module level, which calls PineconeVectorService + YOLODetector).
    #
    # Local-binding patches (e.g. app.services.detection.get_yolo_detector)
    # are needed because 'from X import Y' in production modules creates a
    # local name that is NOT affected by patching 'X.Y' after the module
    # has already been imported.
    with (
        patch("app.services.vector_store.Pinecone", mock_pc_class),
        # ml.* patches keep the lru_cache singleton intact across the session
        patch("ml.yolo_detector.get_yolo_detector", return_value=mock_yolo),
        patch("ml.clip_encoder.get_clip_encoder", return_value=mock_clip),
        # Local-binding patches for route-level service factories
        patch("app.services.detection.get_yolo_detector", return_value=mock_yolo),
        patch("app.services.detect_and_embed.get_yolo_detector", return_value=mock_yolo),
        patch("app.services.detect_and_embed.get_clip_encoder", return_value=mock_clip),
        patch("app.core.pinecone_client.init_pinecone"),
        patch("app.core.pinecone_client.get_pinecone_index", return_value=mock_index),
        patch("app.core.database.engine") as mock_engine,
    ):
        # Mock the async DB engine so lifespan doesn't attempt a real connection
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        begin_cm = AsyncMock()
        begin_cm.__aenter__.return_value = mock_conn
        begin_cm.__aexit__.return_value = False
        mock_engine.begin.return_value = begin_cm
        mock_engine.dispose = AsyncMock()

        from app.main import create_app
        from app.core.dependencies import get_recommendation_service
        import app.api.routes.pipeline as pipeline_module

        app = create_app()

        # Use FastAPI dependency_overrides for the recommendation service
        app.dependency_overrides[get_recommendation_service] = lambda: recommendation_service

        # Replace the module-level pipeline singleton with our stub
        pipeline_module._pipeline = pipeline_stub

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        # Attach stubs so test methods can assert on them
        http_client._recommendation_service = recommendation_service
        http_client._pipeline_stub = pipeline_stub
        try:
            yield http_client
        finally:
            await http_client.aclose()
            app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Health endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")


# ═══════════════════════════════════════════════════════════════════════════════
#  Detection endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectRoute:
    @pytest.mark.asyncio
    async def test_detect_valid_image(self, client):
        resp = await client.post(
            "/api/v1/detect",
            files={"file": ("outfit.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "shirts" in body
        assert "pants" in body
        assert "shoes" in body
        assert body["total_detections"] >= 1

    @pytest.mark.asyncio
    async def test_detect_rejects_pdf(self, client):
        resp = await client.post(
            "/api/v1/detect",
            files={"file": ("doc.pdf", b"fake", "application/pdf")},
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
#  Upload endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadRoute:
    @pytest.mark.asyncio
    async def test_upload_image(self, client):
        resp = await client.post(
            "/api/v1/upload/image",
            files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert "file_path" in resp.json()


class TestRecommendationRoute:
    @pytest.mark.asyncio
    async def test_recommend_returns_results(self, client):
        resp = await client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"budget": "150", "top_n": "5"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_matches"] == 1
        assert body["results"][0]["category"] == "shirt"
        client._recommendation_service.recommend.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recommend_invalid_budget(self, client):
        resp = await client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"budget": "0"},
        )
        assert resp.status_code == 422


class TestPipelineRoute:
    @pytest.mark.asyncio
    async def test_pipeline_recommend_returns_results(self, client):
        resp = await client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("outfit.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_matches"] == 1
        assert body["shirts"][0]["matches"][0]["product_id"] == "prod-1"
        client._pipeline_stub.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_rejects_invalid_file(self, client):
        resp = await client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("bad.txt", b"abc", "text/plain")},
        )
        assert resp.status_code == 400
