"""Unit tests for RecommendationService — the full orchestration layer."""
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import (
    GarmentCategory,
    RecommendationRequest,
)
from app.services.recommendation import RecommendationService
from tests.conftest import make_garment, make_mock_product, make_mock_seller, make_vector_match

PRODUCT_ID = "00000000-0000-0000-0000-000000000001"


def _build_service(
    detect_return=None,
    embed_return=None,
    search_return=None,
    catalog_return=None,
):
    """Wire up a RecommendationService with controllable mock sub-services."""
    detection = MagicMock()
    detection.detect_and_crop = AsyncMock(return_value=detect_return or [])

    embedding = MagicMock()
    embedding.embed = AsyncMock(return_value=embed_return or [])

    search = MagicMock()
    search.search_many = AsyncMock(return_value=search_return or [])

    catalog = MagicMock()
    catalog.get_products_by_ids = AsyncMock(return_value=catalog_return or {})

    return RecommendationService(
        detection=detection,
        embedding=embedding,
        search=search,
        catalog=catalog,
    )


class TestRecommendBasic:
    @pytest.mark.asyncio
    async def test_no_detections_returns_empty(self):
        svc = _build_service()
        request = RecommendationRequest(budget=100.0)
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        assert resp.total_matches == 0
        assert resp.results == []
        assert resp.detected_items == []

    @pytest.mark.asyncio
    async def test_single_detection_within_budget(self):
        garment = make_garment(GarmentCategory.SHIRT)
        crop = Image.new("RGB", (300, 300))
        vec = np.random.rand(512).astype(np.float32)
        match = make_vector_match(product_id=PRODUCT_ID, score=0.90, category=GarmentCategory.SHIRT)
        product = make_mock_product(product_id=PRODUCT_ID, price=50.0, category="shirt")

        svc = _build_service(
            detect_return=[(garment, crop)],
            embed_return=[vec],
            search_return=[match],
            catalog_return={PRODUCT_ID: product},
        )
        request = RecommendationRequest(budget=100.0)
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        assert resp.total_matches == 1
        assert resp.results[0].category == GarmentCategory.SHIRT


class TestBudgetFiltering:
    @pytest.mark.asyncio
    async def test_over_budget_filtered_out(self):
        garment = make_garment(GarmentCategory.SHIRT)
        crop = Image.new("RGB", (300, 300))
        vec = np.random.rand(512).astype(np.float32)
        match = make_vector_match(product_id=PRODUCT_ID, score=0.88, category=GarmentCategory.SHIRT)
        product = make_mock_product(product_id=PRODUCT_ID, price=200.0, category="shirt")

        svc = _build_service(
            detect_return=[(garment, crop)],
            embed_return=[vec],
            search_return=[match],
            catalog_return={PRODUCT_ID: product},
        )
        request = RecommendationRequest(budget=50.0)
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        assert resp.total_matches == 0


class TestLocalityBoost:
    @pytest.mark.asyncio
    async def test_local_seller_gets_boost(self):
        seller = make_mock_seller(latitude=37.77, longitude=-122.42)
        garment = make_garment(GarmentCategory.SHIRT)
        crop = Image.new("RGB", (300, 300))
        vec = np.random.rand(512).astype(np.float32)
        match = make_vector_match(product_id=PRODUCT_ID, score=0.80, category=GarmentCategory.SHIRT)
        product = make_mock_product(product_id=PRODUCT_ID, price=40.0, category="shirt", seller=seller)

        svc = _build_service(
            detect_return=[(garment, crop)],
            embed_return=[vec],
            search_return=[match],
            catalog_return={PRODUCT_ID: product},
        )
        request = RecommendationRequest(
            budget=100.0,
            user_latitude=37.77,
            user_longitude=-122.42,
        )
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        item = resp.results[0].items[0]
        assert item.is_local is True
        # Score should be higher than the raw 0.80 because of locality boost
        assert item.similarity_score > 0.80

    @pytest.mark.asyncio
    async def test_remote_seller_no_boost(self):
        # Seller in London, user in SF — no locality boost
        seller = make_mock_seller(latitude=51.50, longitude=-0.12)
        garment = make_garment(GarmentCategory.SHIRT)
        crop = Image.new("RGB", (300, 300))
        vec = np.random.rand(512).astype(np.float32)
        match = make_vector_match(product_id=PRODUCT_ID, score=0.80, category=GarmentCategory.SHIRT)
        product = make_mock_product(product_id=PRODUCT_ID, price=40.0, category="shirt", seller=seller)

        svc = _build_service(
            detect_return=[(garment, crop)],
            embed_return=[vec],
            search_return=[match],
            catalog_return={PRODUCT_ID: product},
        )
        request = RecommendationRequest(
            budget=100.0,
            user_latitude=37.77,
            user_longitude=-122.42,
        )
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        item = resp.results[0].items[0]
        assert item.is_local is False
        assert item.similarity_score == pytest.approx(0.80, abs=0.01)


class TestMultiCategory:
    @pytest.mark.asyncio
    async def test_multiple_categories_grouped(self):
        """Detections of shirt + pants should produce two GarmentRecommendations."""
        shirt = make_garment(GarmentCategory.SHIRT)
        pants = make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600)
        crop_s = Image.new("RGB", (300, 300))
        crop_p = Image.new("RGB", (260, 240))

        pid_s = str(uuid.uuid4())
        pid_p = str(uuid.uuid4())

        svc = _build_service(
            detect_return=[(shirt, crop_s), (pants, crop_p)],
            embed_return=[np.random.rand(512).astype(np.float32)] * 2,
            search_return=[
                make_vector_match(product_id=pid_s, category=GarmentCategory.SHIRT),
                make_vector_match(product_id=pid_p, category=GarmentCategory.PANTS),
            ],
            catalog_return={
                pid_s: make_mock_product(product_id=pid_s, category="shirt"),
                pid_p: make_mock_product(product_id=pid_p, category="pants"),
            },
        )
        request = RecommendationRequest(budget=500.0)
        resp = await svc.recommend(Image.new("RGB", (640, 640)), request)
        categories = {r.category for r in resp.results}
        assert GarmentCategory.SHIRT in categories
        assert GarmentCategory.PANTS in categories


class TestFilterByBudget:
    def test_static_method_filters(self):
        from app.models.schemas import ProductResponse

        cheap = ProductResponse(
            id=uuid.uuid4(), name="Cheap", category=GarmentCategory.SHIRT,
            price=10.0, currency="USD",
        )
        expensive = ProductResponse(
            id=uuid.uuid4(), name="Expensive", category=GarmentCategory.SHIRT,
            price=500.0, currency="USD",
        )
        result = RecommendationService._filter_by_budget([cheap, expensive], budget=100.0)
        assert len(result) == 1
        assert result[0].name == "Cheap"
