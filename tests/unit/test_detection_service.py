"""Unit tests for DetectionService — mocked YOLO detector throughout."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.models.schemas import (
    ClothingDetectionResponse,
    DetectedGarment,
    GarmentCategory,
)
from app.services.detection import DetectionService

# ═══════════════════════════════════════════════════════════════════════════════
#  detect()
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetect:
    @pytest.mark.asyncio
    async def test_returns_garments(self, sample_image, mock_yolo_detector):
        svc = DetectionService(detector=mock_yolo_detector)
        results = await svc.detect(sample_image)
        assert len(results) == 3
        categories = {r.category for r in results}
        assert GarmentCategory.SHIRT in categories
        assert GarmentCategory.PANTS in categories
        assert GarmentCategory.SHOES in categories

    @pytest.mark.asyncio
    async def test_empty_when_no_detections(self, sample_image):
        detector = MagicMock()
        detector.detect_async = AsyncMock(return_value=[])
        svc = DetectionService(detector=detector)
        assert await svc.detect(sample_image) == []


# ═══════════════════════════════════════════════════════════════════════════════
#  crop()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrop:
    def test_returns_correct_size(self, sample_image, shirt_garment):
        svc = DetectionService(detector=MagicMock())
        crop = svc.crop(sample_image, shirt_garment)
        bb = shirt_garment.bounding_box
        expected_w = int(bb.x_max - bb.x_min)
        expected_h = int(bb.y_max - bb.y_min)
        assert crop.size == (expected_w, expected_h)


# ═══════════════════════════════════════════════════════════════════════════════
#  detect_clothing()
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectClothing:
    @pytest.mark.asyncio
    async def test_groups_by_category(self, sample_image, mock_yolo_detector):
        svc = DetectionService(detector=mock_yolo_detector)
        resp = await svc.detect_clothing(sample_image)
        assert isinstance(resp, ClothingDetectionResponse)
        assert len(resp.shirts) >= 1
        assert resp.total_detections >= 1

    @pytest.mark.asyncio
    async def test_filters_tiny_boxes(self, sample_image, tiny_garment):
        """Bounding boxes < 1 % of image area should be filtered out."""
        detector = MagicMock()
        detector.detect_targets_async = AsyncMock(return_value=[tiny_garment])
        svc = DetectionService(detector=detector)
        resp = await svc.detect_clothing(sample_image)
        assert resp.total_detections == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  detect_and_crop()
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectAndCrop:
    @pytest.mark.asyncio
    async def test_pairs_returned(self, sample_image, mock_yolo_detector):
        svc = DetectionService(detector=mock_yolo_detector)
        pairs = await svc.detect_and_crop(sample_image)
        assert len(pairs) == 3
        for garment, crop in pairs:
            assert isinstance(garment, DetectedGarment)
            assert isinstance(crop, Image.Image)

    @pytest.mark.asyncio
    async def test_tiny_bboxes_filtered(self, sample_image, tiny_garment):
        detector = MagicMock()
        detector.detect_async = AsyncMock(return_value=[tiny_garment])
        svc = DetectionService(detector=detector)
        pairs = await svc.detect_and_crop(sample_image)
        assert pairs == []

    @pytest.mark.asyncio
    async def test_empty_detections(self, sample_image):
        detector = MagicMock()
        detector.detect_async = AsyncMock(return_value=[])
        svc = DetectionService(detector=detector)
        pairs = await svc.detect_and_crop(sample_image)
        assert pairs == []
