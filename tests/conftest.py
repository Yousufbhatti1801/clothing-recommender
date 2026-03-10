"""
Shared fixtures for the clothing-recommender test suite.

Convention
----------
• ML singletons (YOLO, CLIP) are always mocked — tests must never download
  model weights or require a GPU.
• Pinecone is always mocked — no real HTTP calls to the vector DB.
• PostgreSQL can be tested via an in-memory SQLite driver for fast unit tests,
  or a real Postgres container for integration tests (see tests/integration/).
"""
from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image

from app.models.schemas import (
    BoundingBox,
    DetectedGarment,
    GarmentCategory,
    VectorMatch,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Async backend
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ═══════════════════════════════════════════════════════════════════════════════
#  Image helpers
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_image() -> Image.Image:
    """640 × 640 solid-grey image — default for most detection tests."""
    return Image.new("RGB", (640, 640), color=(128, 128, 128))


@pytest.fixture
def small_image() -> Image.Image:
    """100 × 100 image — useful for resize / edge-case tests."""
    return Image.new("RGB", (100, 100), color=(200, 200, 200))


@pytest.fixture
def large_image() -> Image.Image:
    """2048 × 2048 image — triggers resizing logic."""
    return Image.new("RGB", (2048, 2048), color=(64, 64, 64))


@pytest.fixture
def image_bytes() -> bytes:
    """Raw JPEG bytes suitable for UploadFile simulation."""
    buf = BytesIO()
    Image.new("RGB", (640, 640), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  Detection fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def make_garment(
    category: GarmentCategory = GarmentCategory.SHIRT,
    x_min: float = 100,
    y_min: float = 50,
    x_max: float = 400,
    y_max: float = 350,
    confidence: float = 0.92,
) -> DetectedGarment:
    """Factory for a DetectedGarment with sensible defaults."""
    return DetectedGarment(
        category=category,
        bounding_box=BoundingBox(
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, confidence=confidence
        ),
    )


@pytest.fixture
def shirt_garment() -> DetectedGarment:
    return make_garment(GarmentCategory.SHIRT, confidence=0.92)


@pytest.fixture
def pants_garment() -> DetectedGarment:
    return make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600, confidence=0.85)


@pytest.fixture
def shoes_garment() -> DetectedGarment:
    return make_garment(GarmentCategory.SHOES, x_min=150, y_min=610, x_max=350, y_max=640, confidence=0.78)


@pytest.fixture
def tiny_garment() -> DetectedGarment:
    """BB < 1 % of 640×640 — should be filtered out by detect_and_crop."""
    return make_garment(GarmentCategory.SHOES, x_min=0, y_min=0, x_max=5, y_max=5, confidence=0.9)


@pytest.fixture
def all_garments(shirt_garment, pants_garment, shoes_garment) -> list[DetectedGarment]:
    return [shirt_garment, pants_garment, shoes_garment]


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock ML models
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_yolo_detector(all_garments):
    """MagicMock that mimics YOLODetector with async helpers."""
    detector = MagicMock()
    detector.detect_async = AsyncMock(return_value=all_garments)
    detector.detect_targets_async = AsyncMock(return_value=all_garments)
    return detector


@pytest.fixture
def mock_clip_encoder():
    """MagicMock that mimics CLIPEncoder — returns random 512-d vectors."""
    encoder = MagicMock()

    def _encode_side_effect(images):
        return np.random.rand(len(images), 512).astype(np.float32)

    async def _encode_async_side_effect(images):
        return _encode_side_effect(images)

    encoder.encode = MagicMock(side_effect=_encode_side_effect)
    encoder.encode_async = AsyncMock(side_effect=_encode_async_side_effect)
    return encoder


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock services
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_detection_service(mock_yolo_detector):
    """DetectionService wired to the mock YOLO detector."""
    from app.services.detection import DetectionService

    return DetectionService(detector=mock_yolo_detector)


@pytest.fixture
def mock_embedding_service(mock_clip_encoder):
    """EmbeddingService wired to the mock CLIP encoder."""
    from app.services.embedding import EmbeddingService

    return EmbeddingService(encoder=mock_clip_encoder)


@pytest.fixture
def mock_pinecone_index():
    """MagicMock that mimics a pinecone.Index."""
    index = MagicMock()
    index.query = MagicMock(
        return_value={
            "matches": [
                {"id": str(uuid.uuid4()), "score": 0.91, "metadata": {}},
                {"id": str(uuid.uuid4()), "score": 0.85, "metadata": {}},
                {"id": str(uuid.uuid4()), "score": 0.79, "metadata": {}},
            ]
        }
    )
    index.upsert = MagicMock(return_value=None)
    index.describe_index_stats = MagicMock(return_value={"dimension": 512})
    return index


@pytest.fixture
def mock_search_service(mock_pinecone_index):
    from app.services.search import SearchService

    return SearchService(index=mock_pinecone_index)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORM / Product factories
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_product(
    product_id: str | None = None,
    name: str = "Test Item",
    brand: str = "TestBrand",
    category: str = "shirt",
    price: float = 49.99,
    currency: str = "USD",
    seller=None,
):
    """Factory that returns a MagicMock mimicking the Product ORM model."""
    pid = product_id or str(uuid.uuid4())
    product = MagicMock()
    product.id = uuid.UUID(pid)
    product.name = name
    product.brand = brand
    product.category = category
    product.price = price
    product.currency = currency
    product.image_url = "https://example.com/img.jpg"
    product.product_url = None
    product.seller = seller
    product.vector_id = pid
    return product


def make_mock_seller(
    seller_id: str | None = None,
    name: str = "Local Shop",
    city: str = "San Francisco",
    country: str = "US",
    latitude: float = 37.7749,
    longitude: float = -122.4194,
):
    """Factory that returns a MagicMock mimicking the Seller ORM model."""
    sid = seller_id or str(uuid.uuid4())
    seller = MagicMock()
    seller.id = uuid.UUID(sid)
    seller.name = name
    seller.city = city
    seller.country = country
    seller.latitude = latitude
    seller.longitude = longitude
    seller.website = "https://localshop.example.com"
    return seller


def make_vector_match(
    product_id: str | None = None,
    score: float = 0.88,
    category: GarmentCategory = GarmentCategory.SHIRT,
) -> VectorMatch:
    pid = product_id or str(uuid.uuid4())
    return VectorMatch(product_id=pid, score=score, category=category)
