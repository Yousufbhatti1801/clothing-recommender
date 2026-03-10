"""Unit tests for RecommendationPipeline (Pinecone-only, no DB)."""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import GarmentCategory, PipelineRecommendationResponse
from app.services.detect_and_embed import GarmentEmbedding, PipelineResult
from app.services.recommendation_pipeline import RecommendationPipeline
from app.services.vector_store import VectorResult
from tests.conftest import make_garment


def _make_ge(category: GarmentCategory) -> GarmentEmbedding:
    return GarmentEmbedding(
        garment=make_garment(category),
        crop=Image.new("RGB", (100, 100)),
        embedding=np.random.rand(512).astype(np.float32),
    )


class TestRecommendationPipeline:
    @pytest.mark.asyncio
    async def test_run_full_pipeline(self, sample_image):
        ge_shirt = _make_ge(GarmentCategory.SHIRT)
        pipeline_result = PipelineResult(shirts=[ge_shirt])

        detect_embed = MagicMock()
        detect_embed.run = AsyncMock(return_value=pipeline_result)

        vector_store = MagicMock()
        vector_store.query = MagicMock(return_value=[
            VectorResult(id="prod-1", score=0.92, metadata={"name": "Blue Shirt"}),
            VectorResult(id="prod-2", score=0.85, metadata={"name": "Red Shirt"}),
        ])

        pipeline = RecommendationPipeline(
            detect_embed=detect_embed,
            vector_store=vector_store,
            top_k=5,
        )
        resp = await pipeline.run(sample_image)  # type: ignore[arg-type]

        assert isinstance(resp, PipelineRecommendationResponse)
        assert len(resp.shirts) == 1
        assert len(resp.shirts[0].matches) == 2
        assert resp.total_detections == 1
        assert resp.total_matches == 2

    @pytest.mark.asyncio
    async def test_run_no_detections(self, sample_image):
        detect_embed = MagicMock()
        detect_embed.run = AsyncMock(return_value=PipelineResult())

        pipeline = RecommendationPipeline(detect_embed=detect_embed, vector_store=MagicMock())
        resp = await pipeline.run(sample_image)
        assert resp.total_detections == 0
        assert resp.total_matches == 0

    @pytest.mark.asyncio
    async def test_multiple_categories(self, sample_image):
        pipeline_result = PipelineResult(
            shirts=[_make_ge(GarmentCategory.SHIRT)],
            pants=[_make_ge(GarmentCategory.PANTS)],
            shoes=[_make_ge(GarmentCategory.SHOES)],
        )

        detect_embed = MagicMock()
        detect_embed.run = AsyncMock(return_value=pipeline_result)

        vector_store = MagicMock()
        vector_store.query = MagicMock(return_value=[
            VectorResult(id="p-1", score=0.90, metadata={}),
        ])

        pipeline = RecommendationPipeline(
            detect_embed=detect_embed,
            vector_store=vector_store,
        )
        resp = await pipeline.run(sample_image)  # type: ignore[arg-type]
        assert resp.total_detections == 3
        assert len(resp.shirts) == 1
        assert len(resp.pants) == 1
        assert len(resp.shoes) == 1
