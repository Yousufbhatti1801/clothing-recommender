"""Unit tests for DetectAndEmbedPipeline — mocked detection + embedding."""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import GarmentCategory
from app.services.detect_and_embed import (
    DetectAndEmbedPipeline,
    GarmentEmbedding,
    PipelineResult,
)
from tests.conftest import make_garment


class TestPipelineResult:
    def test_total_and_all(self):
        ge = GarmentEmbedding(
            garment=make_garment(GarmentCategory.SHIRT),
            crop=Image.new("RGB", (100, 100)),
            embedding=np.zeros(512, dtype=np.float32),
        )
        result = PipelineResult(shirts=[ge])
        assert result.total == 1
        assert len(result.all) == 1

    def test_empty_result(self):
        result = PipelineResult()
        assert result.total == 0
        assert result.all == []


class TestGarmentEmbedding:
    def test_properties(self):
        garment = make_garment(GarmentCategory.PANTS, confidence=0.85)
        ge = GarmentEmbedding(
            garment=garment,
            crop=Image.new("RGB", (100, 100)),
            embedding=np.zeros(512, dtype=np.float32),
        )
        assert ge.category == GarmentCategory.PANTS
        assert ge.confidence == 0.85


class TestDetectAndEmbedPipeline:
    @pytest.mark.asyncio
    async def test_run_with_detections(self, sample_image):
        shirt = make_garment(GarmentCategory.SHIRT)
        pants = make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600)

        # Mock detection service
        detection = MagicMock()
        detection.detect_and_crop = AsyncMock(return_value=[
            (shirt, Image.new("RGB", (300, 300))),
            (pants, Image.new("RGB", (260, 240))),
        ])

        # Mock embedding service
        embedding = MagicMock()
        embedding.embed = AsyncMock(return_value=[
            np.random.rand(512).astype(np.float32),
            np.random.rand(512).astype(np.float32),
        ])

        pipeline = DetectAndEmbedPipeline(
            detection_service=detection,
            embedding_service=embedding,
        )
        result = await pipeline.run(sample_image)

        assert isinstance(result, PipelineResult)
        assert result.total == 2
        assert len(result.shirts) == 1
        assert len(result.pants) == 1
        assert len(result.shoes) == 0

    @pytest.mark.asyncio
    async def test_run_no_detections(self, sample_image):
        detection = MagicMock()
        detection.detect_and_crop = AsyncMock(return_value=[])
        embedding = MagicMock()

        pipeline = DetectAndEmbedPipeline(
            detection_service=detection,
            embedding_service=embedding,
        )
        result = await pipeline.run(sample_image)
        assert result.total == 0
        embedding.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_embedding_shape_correct(self, sample_image):
        garment = make_garment(GarmentCategory.SHOES, x_min=150, y_min=610, x_max=350, y_max=640)
        detection = MagicMock()
        detection.detect_and_crop = AsyncMock(return_value=[
            (garment, Image.new("RGB", (200, 30))),
        ])
        vec = np.random.rand(512).astype(np.float32)
        embedding = MagicMock()
        embedding.embed = AsyncMock(return_value=[vec])

        pipeline = DetectAndEmbedPipeline(
            detection_service=detection,
            embedding_service=embedding,
        )
        result = await pipeline.run(sample_image)
        assert result.shoes[0].embedding.shape == (512,)
