"""Unit tests for IngestionService — mocked catalog, embedding, Pinecone, HTTP."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import GarmentCategory, ProductIngestRequest
from app.services.ingestion import IngestionService
from tests.conftest import make_mock_product


@pytest.fixture
def ingest_request() -> ProductIngestRequest:
    return ProductIngestRequest(
        name="White Sneakers",
        brand="Nike",
        category=GarmentCategory.SHOES,
        price=89.99,
        image_url="https://example.com/sneakers.jpg",
    )


@pytest.fixture
def mock_catalog():
    svc = AsyncMock()
    product = make_mock_product(category="shoes", price=89.99, name="White Sneakers")
    svc.create_product.return_value = product
    svc.set_vector_id = AsyncMock()
    return svc


@pytest.fixture
def mock_embedding():
    svc = AsyncMock()
    svc.embed_single = AsyncMock(return_value=np.random.rand(512).astype(np.float32))
    return svc


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.upsert = MagicMock()
    return index


class TestIngest:
    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_full_flow(
        self, mock_download, ingest_request, mock_catalog, mock_embedding, mock_index
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))

        svc = IngestionService(
            catalog=mock_catalog, embedding=mock_embedding, index=mock_index
        )
        resp = await svc.ingest(ingest_request)

        # Checks the whole pipeline executed
        mock_download.assert_awaited_once()
        mock_embedding.embed_single.assert_awaited_once()
        mock_catalog.create_product.assert_awaited_once()
        mock_index.upsert.assert_called_once()
        mock_catalog.set_vector_id.assert_awaited_once()

        assert resp.vector_id is not None
        assert resp.message == "Product indexed successfully"

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_upsert_uses_correct_namespace(
        self, mock_download, ingest_request, mock_catalog, mock_embedding, mock_index
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))

        svc = IngestionService(
            catalog=mock_catalog, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request)

        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["namespace"] == "shoes"
