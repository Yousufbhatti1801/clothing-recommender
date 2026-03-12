"""
tests/unit/test_ingestion_metadata.py
======================================
Verify that IngestionService upserts rich metadata to Pinecone and stores
the vector_id back in the database via CatalogService.set_vector_id.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import GarmentCategory, ProductIngestRequest
from app.services.ingestion import IngestionService
from tests.conftest import make_mock_product

# ── Shared helpers ────────────────────────────────────────────────────────────

CATEGORY = GarmentCategory.SHOES
SELLER_ID = uuid.uuid4()


@pytest.fixture
def ingest_request_full() -> ProductIngestRequest:
    """Request with every optional field populated."""
    return ProductIngestRequest(
        name="Classic White Sneakers",
        brand="PureBrand",
        category=CATEGORY,
        price=89.99,
        currency="USD",
        image_url="https://images.unsplash.com/photo-abc?w=640",
        seller_id=SELLER_ID,
    )


@pytest.fixture
def ingest_request_minimal() -> ProductIngestRequest:
    """Request with optional brand and seller_id omitted."""
    return ProductIngestRequest(
        name="Plain Shoe",
        category=CATEGORY,
        price=49.99,
        image_url="https://images.unsplash.com/photo-xyz?w=640",
    )


@pytest.fixture
def mock_catalog_full():
    svc = AsyncMock()
    product = make_mock_product(
        category="shoes",
        price=89.99,
        name="Classic White Sneakers",
        seller=None,
    )
    svc.create_product.return_value = product
    svc.set_vector_id = AsyncMock()
    return svc


@pytest.fixture
def mock_catalog_minimal():
    svc = AsyncMock()
    product = make_mock_product(
        category="shoes",
        price=49.99,
        name="Plain Shoe",
        seller=None,
    )
    svc.create_product.return_value = product
    svc.set_vector_id = AsyncMock()
    return svc


@pytest.fixture
def mock_embedding():
    svc = AsyncMock()
    svc.embed_single = AsyncMock(
        return_value=np.random.default_rng(0).random(512).astype(np.float32)
    )
    return svc


@pytest.fixture
def mock_index():
    idx = MagicMock()
    idx.upsert = MagicMock()
    return idx


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMetadataShape:
    """The metadata dict passed to Pinecone must contain all required keys."""

    REQUIRED_KEYS = {"price", "currency", "brand", "category", "name", "seller_id", "image_url"}

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_all_required_keys_present(
        self,
        mock_download,
        ingest_request_full,
        mock_catalog_full,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))

        svc = IngestionService(
            catalog=mock_catalog_full,
            embedding=mock_embedding,
            index=mock_index,
        )
        await svc.ingest(ingest_request_full)

        vectors_kwarg = mock_index.upsert.call_args.kwargs["vectors"]
        assert len(vectors_kwarg) == 1
        metadata = vectors_kwarg[0]["metadata"]
        missing = self.REQUIRED_KEYS - metadata.keys()
        assert not missing, f"Missing metadata keys: {missing}"

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_price_is_float(
        self,
        mock_download,
        ingest_request_full,
        mock_catalog_full,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))
        svc = IngestionService(
            catalog=mock_catalog_full, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request_full)

        meta = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert isinstance(meta["price"], float)
        assert meta["price"] == pytest.approx(89.99)

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_category_stored_as_string_value(
        self,
        mock_download,
        ingest_request_full,
        mock_catalog_full,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))
        svc = IngestionService(
            catalog=mock_catalog_full, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request_full)

        meta = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert meta["category"] == "shoes"

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_namespace_matches_category(
        self,
        mock_download,
        ingest_request_full,
        mock_catalog_full,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))
        svc = IngestionService(
            catalog=mock_catalog_full, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request_full)

        ns = mock_index.upsert.call_args.kwargs["namespace"]
        meta = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert ns == meta["category"] == "shoes"


class TestNoneFieldHandling:
    """None / missing optional fields must be serialised safely (never raise)."""

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_missing_brand_becomes_empty_string(
        self,
        mock_download,
        ingest_request_minimal,
        mock_catalog_minimal,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))
        svc = IngestionService(
            catalog=mock_catalog_minimal, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request_minimal)

        meta = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert meta["brand"] == ""

    @pytest.mark.asyncio
    @patch("app.services.ingestion.IngestionService._download_image")
    async def test_missing_seller_id_becomes_empty_string(
        self,
        mock_download,
        ingest_request_minimal,
        mock_catalog_minimal,
        mock_embedding,
        mock_index,
    ):
        mock_download.return_value = Image.new("RGB", (224, 224))
        svc = IngestionService(
            catalog=mock_catalog_minimal, embedding=mock_embedding, index=mock_index
        )
        await svc.ingest(ingest_request_minimal)

        meta = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert meta["seller_id"] == ""
