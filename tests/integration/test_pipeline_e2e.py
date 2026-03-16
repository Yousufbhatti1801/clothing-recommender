"""Integration test: detect → embed → search pipeline end-to-end (mocked externals)."""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.models.schemas import GarmentCategory
from app.services.detect_and_embed import DetectAndEmbedPipeline
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.recommendation_pipeline import RecommendationPipeline
from app.services.vector_store import VectorResult
from tests.conftest import make_garment


class TestEndToEndPipeline:
    """
    Exercises the full path: image → YOLO detect → crop → CLIP embed → Pinecone
    search, with ML models and Pinecone mocked.

    This verifies that all services are correctly wired together.
    """

    @pytest.mark.asyncio
    async def test_full_flow_returns_results(self, sample_image):
        # ── Mock YOLO ────────────────────────────────────────────────────
        shirt = make_garment(GarmentCategory.SHIRT)
        pants = make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600)

        yolo = MagicMock()
        yolo.detect_async = AsyncMock(return_value=[shirt, pants])

        # ── Mock CLIP ────────────────────────────────────────────────────
        clip = MagicMock()
        clip.encode_async = AsyncMock(
            return_value=np.random.rand(2, 512).astype(np.float32)
        )

        # ── Wire services ────────────────────────────────────────────────
        detection_svc = DetectionService(detector=yolo)
        embedding_svc = EmbeddingService(encoder=clip)

        detect_embed = DetectAndEmbedPipeline(
            detection_service=detection_svc,
            embedding_service=embedding_svc,
        )

        # ── Mock Pinecone ────────────────────────────────────────────────
        vector_store = MagicMock()
        vector_store.query = MagicMock(return_value=[
            VectorResult(id="p-1", score=0.93, metadata={}),
            VectorResult(id="p-2", score=0.87, metadata={}),
        ])

        pipeline = RecommendationPipeline(
            detect_embed=detect_embed,
            vector_store=vector_store,
            top_k=5,
        )

        # ── Execute ──────────────────────────────────────────────────────
        resp = await pipeline.run(sample_image)

        assert resp.total_detections == 2
        assert resp.total_matches > 0
        assert len(resp.shirts) >= 1
        assert len(resp.pants) >= 1

    @pytest.mark.asyncio
    async def test_no_detections_empty_response(self, sample_image):
        yolo = MagicMock()
        yolo.detect_async = AsyncMock(return_value=[])

        clip = MagicMock()

        detection_svc = DetectionService(detector=yolo)
        embedding_svc = EmbeddingService(encoder=clip)

        detect_embed = DetectAndEmbedPipeline(
            detection_service=detection_svc,
            embedding_service=embedding_svc,
        )

        pipeline = RecommendationPipeline(detect_embed=detect_embed, vector_store=MagicMock())
        resp = await pipeline.run(sample_image)

        assert resp.total_detections == 0
        assert resp.total_matches == 0

    @pytest.mark.asyncio
    async def test_detect_and_embed_produces_correct_shapes(self, sample_image):
        """Verify intermediate embeddings have the correct shape."""
        garment = make_garment(GarmentCategory.SHOES, x_min=150, y_min=610, x_max=350, y_max=640)

        yolo = MagicMock()
        yolo.detect_async = AsyncMock(return_value=[garment])

        clip = MagicMock()
        clip.encode_async = AsyncMock(
            return_value=np.random.rand(1, 512).astype(np.float32)
        )

        detection_svc = DetectionService(detector=yolo)
        embedding_svc = EmbeddingService(encoder=clip)

        pipeline = DetectAndEmbedPipeline(
            detection_service=detection_svc,
            embedding_service=embedding_svc,
        )
        result = await pipeline.run(sample_image)

        assert result.total == 1
        assert result.shoes[0].embedding.shape == (512,)
        assert result.shoes[0].confidence == garment.bounding_box.confidence
