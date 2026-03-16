from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.models.schemas import BoundingBox, DetectedGarment, GarmentCategory
from app.services.detection import DetectionService


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (640, 640), color=(128, 128, 128))


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.detect_async = AsyncMock(
        return_value=[
            DetectedGarment(
                category=GarmentCategory.SHIRT,
                bounding_box=BoundingBox(
                    x_min=100, y_min=50, x_max=400, y_max=350, confidence=0.92
                ),
            ),
            DetectedGarment(
                category=GarmentCategory.PANTS,
                bounding_box=BoundingBox(
                    x_min=120, y_min=360, x_max=380, y_max=600, confidence=0.85
                ),
            ),
        ]
    )
    return detector


@pytest.mark.asyncio
async def test_detect_returns_garments(sample_image, mock_detector):
    service = DetectionService(detector=mock_detector)
    results = await service.detect(sample_image)
    assert len(results) == 2
    categories = {r.category for r in results}
    assert GarmentCategory.SHIRT in categories
    assert GarmentCategory.PANTS in categories


@pytest.mark.asyncio
async def test_detect_and_crop_filters_tiny_boxes(sample_image):
    """Bounding boxes < 1 % of image area should be filtered out."""
    tiny_detector = MagicMock()
    tiny_detector.detect_async = AsyncMock(
        return_value=[
            DetectedGarment(
                category=GarmentCategory.SHOES,
                bounding_box=BoundingBox(
                    x_min=0, y_min=0, x_max=5, y_max=5, confidence=0.9  # tiny
                ),
            )
        ]
    )
    service = DetectionService(detector=tiny_detector)
    results = await service.detect_and_crop(sample_image)
    assert results == []


@pytest.mark.asyncio
async def test_crop_returns_correct_size(sample_image, mock_detector):
    service = DetectionService(detector=mock_detector)
    pairs = await service.detect_and_crop(sample_image)
    assert len(pairs) == 2
    garment, crop = pairs[0]
    bb = garment.bounding_box
    expected_w = int(bb.x_max - bb.x_min)
    expected_h = int(bb.y_max - bb.y_min)
    assert crop.size == (expected_w, expected_h)
