import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import (
    BoundingBox,
    DetectedGarment,
    GarmentCategory,
    RecommendationRequest,
    VectorMatch,
)
from app.services.recommendation import RecommendationService


def _make_mock_product(product_id: str, price: float, category: str, seller=None):
    p = MagicMock()
    p.id = uuid.UUID(product_id)
    p.name = "Test Item"
    p.brand = "TestBrand"
    p.category = category
    p.price = price
    p.currency = "USD"
    p.image_url = "https://example.com/img.jpg"
    p.product_url = None
    p.seller = seller
    return p


PRODUCT_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def mock_detection():
    svc = MagicMock()
    svc.detect_and_crop = AsyncMock(
        return_value=[
            (
                DetectedGarment(
                    category=GarmentCategory.SHIRT,
                    bounding_box=BoundingBox(
                        x_min=10, y_min=10, x_max=200, y_max=300, confidence=0.9
                    ),
                ),
                Image.new("RGB", (190, 290)),
            )
        ]
    )
    return svc


@pytest.fixture
def mock_embedding():
    svc = MagicMock()
    svc.embed = AsyncMock(return_value=[np.random.rand(512).astype(np.float32)])
    return svc


@pytest.fixture
def mock_search():
    svc = MagicMock()
    svc.search_many = AsyncMock(
        return_value=[VectorMatch(product_id=PRODUCT_ID, score=0.87, category=GarmentCategory.SHIRT)]
    )
    return svc


@pytest.fixture
def mock_catalog():
    svc = MagicMock()
    svc.get_products_by_ids = AsyncMock(
        return_value={PRODUCT_ID: _make_mock_product(PRODUCT_ID, 50.0, "shirt")}
    )
    return svc


@pytest.mark.asyncio
async def test_recommend_within_budget(mock_detection, mock_embedding, mock_search, mock_catalog):
    service = RecommendationService(
        detection=mock_detection,
        embedding=mock_embedding,
        search=mock_search,
        catalog=mock_catalog,
    )
    request = RecommendationRequest(budget=100.0, top_n=5)
    response = await service.recommend(Image.new("RGB", (640, 640)), request)
    assert response.total_matches == 1
    assert response.results[0].items[0].similarity_score == pytest.approx(0.87, abs=0.01)


@pytest.mark.asyncio
async def test_recommend_over_budget_filtered(mock_detection, mock_embedding, mock_search, mock_catalog):
    mock_catalog.get_products_by_ids = AsyncMock(
        return_value={PRODUCT_ID: _make_mock_product(PRODUCT_ID, 200.0, "shirt")}
    )
    service = RecommendationService(
        detection=mock_detection,
        embedding=mock_embedding,
        search=mock_search,
        catalog=mock_catalog,
    )
    request = RecommendationRequest(budget=50.0, top_n=5)
    response = await service.recommend(Image.new("RGB", (640, 640)), request)
    assert response.total_matches == 0


@pytest.mark.asyncio
async def test_recommend_no_detections(mock_embedding, mock_search, mock_catalog):
    no_detect = MagicMock()
    no_detect.detect_and_crop = AsyncMock(return_value=[])
    service = RecommendationService(
        detection=no_detect,
        embedding=mock_embedding,
        search=mock_search,
        catalog=mock_catalog,
    )
    request = RecommendationRequest(budget=100.0)
    response = await service.recommend(Image.new("RGB", (640, 640)), request)
    assert response.total_matches == 0
    assert response.detected_items == []
