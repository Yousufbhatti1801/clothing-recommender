"""Unit tests for EmbeddingService — mocked CLIP encoder."""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.embedding import EmbeddingService


class TestEmbed:
    @pytest.mark.asyncio
    async def test_returns_correct_count(self, mock_clip_encoder):
        crops = [Image.new("RGB", (224, 224)) for _ in range(4)]
        svc = EmbeddingService(encoder=mock_clip_encoder)
        results = await svc.embed(crops)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_returns_512d_float32(self, mock_clip_encoder):
        svc = EmbeddingService(encoder=mock_clip_encoder)
        results = await svc.embed([Image.new("RGB", (224, 224))])
        assert results[0].shape == (512,)
        assert results[0].dtype == np.float32

    @pytest.mark.asyncio
    async def test_empty_list_skips_encoder(self, mock_clip_encoder):
        svc = EmbeddingService(encoder=mock_clip_encoder)
        results = await svc.embed([])
        assert results == []
        mock_clip_encoder.encode_async.assert_not_called()


class TestEmbedSingle:
    @pytest.mark.asyncio
    async def test_single_image(self, mock_clip_encoder):
        svc = EmbeddingService(encoder=mock_clip_encoder)
        vec = await svc.embed_single(Image.new("RGB", (224, 224)))
        assert vec.shape == (512,)
        assert vec.dtype == np.float32
