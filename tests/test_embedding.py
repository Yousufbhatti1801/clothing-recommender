from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.embedding import EmbeddingService


@pytest.fixture
def crops() -> list[Image.Image]:
    return [Image.new("RGB", (224, 224)) for _ in range(3)]


@pytest.fixture
def mock_encoder():
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(
        return_value=np.random.rand(3, 512).astype(np.float32)
    )
    return encoder


@pytest.mark.asyncio
async def test_embed_returns_correct_count(crops, mock_encoder):
    service = EmbeddingService(encoder=mock_encoder)
    results = await service.embed(crops)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_embed_returns_float32_vectors(crops, mock_encoder):
    service = EmbeddingService(encoder=mock_encoder)
    results = await service.embed(crops)
    for vec in results:
        assert vec.dtype == np.float32
        assert vec.shape == (512,)


@pytest.mark.asyncio
async def test_embed_empty_list(mock_encoder):
    service = EmbeddingService(encoder=mock_encoder)
    results = await service.embed([])
    assert results == []
    mock_encoder.encode_async.assert_not_called()


@pytest.mark.asyncio
async def test_embed_single(mock_encoder):
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(
        return_value=np.random.rand(1, 512).astype(np.float32)
    )
    service = EmbeddingService(encoder=encoder)
    result = await service.embed_single(Image.new("RGB", (224, 224)))
    assert result.shape == (512,)
