"""
tests/test_fashion_detection.py
================================
Validate the fashion detection pipeline end-to-end:
  • Detection returns valid DetectedGarment objects
  • Bounding boxes are well-formed
  • Categories map correctly
  • Multi-garment detection works
  • Crop pipeline integrates correctly
  • Pipeline integration (detect → crop → embed)

Uses both mocked detectors (for deterministic tests) and the real model
(for smoke tests marked as @pytest.mark.slow).
"""
from __future__ import annotations

import os
import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image

os.environ.setdefault("PINECONE_API_KEY", "dummy")

from app.models.schemas import (
    BoundingBox,
    DetectedGarment,
    GarmentCategory,
)
from ml.fashion_classes import APP_CLASS_NAMES, build_label_map_from_model_names


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_garment(
    category: GarmentCategory = GarmentCategory.SHIRT,
    x_min: float = 100, y_min: float = 50,
    x_max: float = 400, y_max: float = 350,
    confidence: float = 0.92,
) -> DetectedGarment:
    return DetectedGarment(
        category=category,
        bounding_box=BoundingBox(
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            confidence=confidence,
        ),
    )


def _make_sample_image(w: int = 640, h: int = 640) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 128, 128))


# ══════════════════════════════════════════════════════════════════════════════
# Test class 1: Detection output validation
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionOutputValidation:
    """Ensure detected garments have valid structure."""

    def test_bounding_box_coordinates_valid(self):
        g = _make_garment(x_min=10, y_min=20, x_max=300, y_max=400)
        bb = g.bounding_box
        assert bb.x_min < bb.x_max
        assert bb.y_min < bb.y_max
        assert bb.confidence > 0
        assert bb.confidence <= 1

    def test_garment_has_category(self):
        g = _make_garment(GarmentCategory.PANTS)
        assert g.category == GarmentCategory.PANTS
        assert isinstance(g.category, GarmentCategory)

    def test_all_categories_creatable(self):
        for cat in GarmentCategory:
            g = _make_garment(cat)
            assert g.category == cat

    def test_bounding_box_area_positive(self):
        g = _make_garment(x_min=50, y_min=50, x_max=200, y_max=300)
        bb = g.bounding_box
        area = (bb.x_max - bb.x_min) * (bb.y_max - bb.y_min)
        assert area > 0


# ══════════════════════════════════════════════════════════════════════════════
# Test class 2: Multi-garment detection (mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiGarmentDetection:
    """Verify multi-garment detection with mocked YOLO results."""

    @pytest.fixture
    def mock_detector(self):
        """Mock YOLODetector that returns shirt + pants + shoes."""
        import warnings
        warnings.filterwarnings("ignore")
        from ml.yolo_detector import YOLODetector

        detector = MagicMock(spec=YOLODetector)
        detector.is_fashion_model = True
        detector.confidence_threshold = 0.4
        detector.model_class_names = {i: APP_CLASS_NAMES[i] for i in range(13)}

        detections = [
            _make_garment(GarmentCategory.SHIRT, 100, 50, 400, 280, 0.92),
            _make_garment(GarmentCategory.PANTS, 120, 300, 380, 550, 0.87),
            _make_garment(GarmentCategory.SHOES, 150, 570, 350, 630, 0.79),
        ]
        detector.detect.return_value = detections
        detector.detect_targets.return_value = detections
        detector.detect_all_fashion.return_value = detections
        detector.detect_async = AsyncMock(return_value=detections)
        detector.detect_targets_async = AsyncMock(return_value=detections)
        return detector

    def test_detects_three_garments(self, mock_detector):
        img = _make_sample_image()
        result = mock_detector.detect(img)
        assert len(result) == 3

    def test_shirt_detected(self, mock_detector):
        result = mock_detector.detect(_make_sample_image())
        cats = [d.category for d in result]
        assert GarmentCategory.SHIRT in cats

    def test_pants_detected(self, mock_detector):
        result = mock_detector.detect(_make_sample_image())
        cats = [d.category for d in result]
        assert GarmentCategory.PANTS in cats

    def test_shoes_detected(self, mock_detector):
        result = mock_detector.detect(_make_sample_image())
        cats = [d.category for d in result]
        assert GarmentCategory.SHOES in cats

    def test_detect_targets_returns_target_categories(self, mock_detector):
        from ml.yolo_detector import TARGET_CATEGORIES
        result = mock_detector.detect_targets(_make_sample_image())
        for d in result:
            assert d.category in TARGET_CATEGORIES

    def test_all_fashion_includes_extended(self, mock_detector):
        # Add jacket + dress to the mock
        extended = [
            _make_garment(GarmentCategory.JACKET, 80, 30, 420, 260, 0.85),
            _make_garment(GarmentCategory.DRESS, 100, 40, 400, 500, 0.80),
        ]
        mock_detector.detect_all_fashion.return_value = extended
        result = mock_detector.detect_all_fashion(_make_sample_image())
        cats = {d.category for d in result}
        assert GarmentCategory.JACKET in cats
        assert GarmentCategory.DRESS in cats


# ══════════════════════════════════════════════════════════════════════════════
# Test class 3: Crop pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestCropPipeline:
    """Verify garment crop extraction from detected bounding boxes."""

    def test_crop_correct_dimensions(self):
        img = _make_sample_image(640, 640)
        garment = _make_garment(x_min=100, y_min=50, x_max=400, y_max=350)
        bb = garment.bounding_box
        crop = img.crop((bb.x_min, bb.y_min, bb.x_max, bb.y_max))
        assert crop.width == 300  # 400 - 100
        assert crop.height == 300  # 350 - 50

    def test_crop_is_rgb(self):
        img = _make_sample_image()
        garment = _make_garment()
        bb = garment.bounding_box
        crop = img.crop((bb.x_min, bb.y_min, bb.x_max, bb.y_max))
        assert crop.mode == "RGB"

    def test_crop_multiple_garments(self):
        img = _make_sample_image()
        garments = [
            _make_garment(GarmentCategory.SHIRT, 10, 10, 300, 250),
            _make_garment(GarmentCategory.PANTS, 10, 260, 300, 500),
            _make_garment(GarmentCategory.SHOES, 10, 510, 300, 630),
        ]
        crops = []
        for g in garments:
            bb = g.bounding_box
            crop = img.crop((bb.x_min, bb.y_min, bb.x_max, bb.y_max))
            crops.append(crop)
        assert len(crops) == 3
        assert all(c.width > 0 and c.height > 0 for c in crops)

    def test_crop_preserves_area(self):
        img = _make_sample_image()
        garment = _make_garment(x_min=50, y_min=50, x_max=250, y_max=350)
        bb = garment.bounding_box
        expected_w = int(bb.x_max - bb.x_min)
        expected_h = int(bb.y_max - bb.y_min)
        crop = img.crop((bb.x_min, bb.y_min, bb.x_max, bb.y_max))
        assert crop.width == expected_w
        assert crop.height == expected_h


# ══════════════════════════════════════════════════════════════════════════════
# Test class 4: DetectionService integration
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionServiceIntegration:
    """Verify DetectionService correctly wires to detector and crops."""

    @pytest.fixture
    def detection_service_with_mock(self):
        """DetectionService with mocked YOLO returning 3 garments."""
        import warnings
        warnings.filterwarnings("ignore")
        from ml.yolo_detector import YOLODetector

        mock_det = MagicMock(spec=YOLODetector)
        detections = [
            _make_garment(GarmentCategory.SHIRT, 100, 50, 400, 280, 0.92),
            _make_garment(GarmentCategory.PANTS, 120, 300, 380, 550, 0.87),
            _make_garment(GarmentCategory.SHOES, 150, 570, 350, 630, 0.79),
        ]
        mock_det.detect_async = AsyncMock(return_value=detections)
        mock_det.detect_targets_async = AsyncMock(return_value=detections)

        from app.services.detection import DetectionService
        return DetectionService(detector=mock_det)

    async def test_detect_returns_garments(self, detection_service_with_mock):
        img = _make_sample_image()
        result = await detection_service_with_mock.detect(img)
        assert len(result) == 3

    async def test_detect_and_crop_returns_pairs(self, detection_service_with_mock):
        img = _make_sample_image()
        pairs = await detection_service_with_mock.detect_and_crop(img)
        assert len(pairs) == 3
        for garment, crop in pairs:
            assert isinstance(garment, DetectedGarment)
            assert isinstance(crop, Image.Image)

    async def test_crop_dimensions_match_bbox(self, detection_service_with_mock):
        img = _make_sample_image()
        pairs = await detection_service_with_mock.detect_and_crop(img)
        for garment, crop in pairs:
            bb = garment.bounding_box
            expected_w = int(bb.x_max - bb.x_min)
            expected_h = int(bb.y_max - bb.y_min)
            assert crop.width == expected_w
            assert crop.height == expected_h

    async def test_tiny_boxes_filtered(self, detection_service_with_mock):
        """Detection service should filter boxes < 1% of image area."""
        from ml.yolo_detector import YOLODetector
        from app.services.detection import DetectionService

        mock_det = MagicMock(spec=YOLODetector)
        tiny = _make_garment(GarmentCategory.SHOES, 0, 0, 5, 5, 0.9)  # tiny
        normal = _make_garment(GarmentCategory.SHIRT, 100, 50, 400, 350, 0.92)
        mock_det.detect_async = AsyncMock(return_value=[tiny, normal])

        svc = DetectionService(detector=mock_det)
        pairs = await svc.detect_and_crop(_make_sample_image())
        # Only the normal-sized garment should survive
        assert len(pairs) == 1
        assert pairs[0][0].category == GarmentCategory.SHIRT


# ══════════════════════════════════════════════════════════════════════════════
# Test class 5: Pipeline integration (detect → crop → embed)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Verify the full detect → crop → embed pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Set up DetectAndEmbedPipeline with mocked YOLO + CLIP."""
        import warnings
        warnings.filterwarnings("ignore")
        from ml.yolo_detector import YOLODetector
        from ml.clip_encoder import CLIPEncoder
        from app.services.detection import DetectionService
        from app.services.embedding import EmbeddingService
        from app.services.detect_and_embed import DetectAndEmbedPipeline

        # Mock YOLO
        mock_det = MagicMock(spec=YOLODetector)
        detections = [
            _make_garment(GarmentCategory.SHIRT, 100, 50, 400, 280, 0.92),
            _make_garment(GarmentCategory.PANTS, 120, 300, 380, 550, 0.87),
            _make_garment(GarmentCategory.SHOES, 150, 570, 350, 630, 0.79),
        ]
        mock_det.detect_async = AsyncMock(return_value=detections)
        mock_det.detect_targets_async = AsyncMock(return_value=detections)

        # Mock CLIP
        mock_clip = MagicMock(spec=CLIPEncoder)
        async def _encode(images):
            return np.random.rand(len(images), 512).astype(np.float32)
        mock_clip.encode_async = AsyncMock(side_effect=_encode)

        det_svc = DetectionService(detector=mock_det)
        emb_svc = EmbeddingService(encoder=mock_clip)
        return DetectAndEmbedPipeline(
            detection_service=det_svc,
            embedding_service=emb_svc,
        )

    async def test_pipeline_returns_result(self, mock_pipeline):
        from app.services.detect_and_embed import PipelineResult
        img = _make_sample_image()
        result = await mock_pipeline.run(img)
        assert isinstance(result, PipelineResult)

    async def test_pipeline_finds_all_categories(self, mock_pipeline):
        img = _make_sample_image()
        result = await mock_pipeline.run(img)
        assert len(result.shirts) >= 1
        assert len(result.pants) >= 1
        assert len(result.shoes) >= 1

    async def test_pipeline_total_count(self, mock_pipeline):
        img = _make_sample_image()
        result = await mock_pipeline.run(img)
        assert result.total == 3

    async def test_pipeline_embeddings_shape(self, mock_pipeline):
        img = _make_sample_image()
        result = await mock_pipeline.run(img)
        for ge in result.all:
            assert ge.embedding.shape == (512,)
            assert ge.embedding.dtype == np.float32

    async def test_pipeline_garment_has_crop(self, mock_pipeline):
        img = _make_sample_image()
        result = await mock_pipeline.run(img)
        for ge in result.all:
            assert isinstance(ge.crop, Image.Image)
            assert ge.crop.width > 0
            assert ge.crop.height > 0

    async def test_pipeline_empty_image(self, mock_pipeline):
        """Pipeline should handle images with no detections gracefully."""
        from ml.yolo_detector import YOLODetector
        from ml.clip_encoder import CLIPEncoder
        from app.services.detection import DetectionService
        from app.services.embedding import EmbeddingService
        from app.services.detect_and_embed import DetectAndEmbedPipeline

        mock_det = MagicMock(spec=YOLODetector)
        mock_det.detect_async = AsyncMock(return_value=[])
        mock_det.detect_targets_async = AsyncMock(return_value=[])

        mock_clip = MagicMock(spec=CLIPEncoder)

        det_svc = DetectionService(detector=mock_det)
        emb_svc = EmbeddingService(encoder=mock_clip)
        pipeline = DetectAndEmbedPipeline(detection_service=det_svc, embedding_service=emb_svc)

        result = await pipeline.run(_make_sample_image())
        assert result.total == 0
        assert result.shirts == []
        assert result.pants == []
        assert result.shoes == []


# ══════════════════════════════════════════════════════════════════════════════
# Test class 6: Real model smoke test (slow)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestRealModelDetection:
    """Smoke tests using the actual trained model.
    These verify the model runs without errors on various inputs."""

    @pytest.fixture(scope="class")
    def real_detector(self):
        import warnings
        warnings.filterwarnings("ignore")
        from ml.yolo_detector import YOLODetector
        return YOLODetector()

    def test_detect_on_blank_image(self, real_detector):
        img = _make_sample_image()
        result = real_detector.detect(img)
        assert isinstance(result, list)

    def test_detect_on_small_image(self, real_detector):
        img = _make_sample_image(100, 100)
        result = real_detector.detect(img)
        assert isinstance(result, list)

    def test_detect_on_large_image(self, real_detector):
        img = _make_sample_image(2048, 2048)
        result = real_detector.detect(img)
        assert isinstance(result, list)

    def test_detect_targets_filters_categories(self, real_detector):
        from ml.yolo_detector import TARGET_CATEGORIES
        img = _make_sample_image()
        result = real_detector.detect_targets(img)
        for d in result:
            assert d.category in TARGET_CATEGORIES

    def test_detect_all_fashion_filters_categories(self, real_detector):
        from ml.yolo_detector import ALL_FASHION_CATEGORIES
        img = _make_sample_image()
        result = real_detector.detect_all_fashion(img)
        for d in result:
            assert d.category in ALL_FASHION_CATEGORIES

    def test_detections_sorted_by_confidence(self, real_detector):
        img = _make_sample_image()
        result = real_detector.detect(img)
        if len(result) >= 2:
            confidences = [d.bounding_box.confidence for d in result]
            assert confidences == sorted(confidences, reverse=True)

    def test_detections_have_valid_boxes(self, real_detector):
        img = _make_sample_image()
        result = real_detector.detect(img)
        for d in result:
            bb = d.bounding_box
            assert bb.x_min < bb.x_max
            assert bb.y_min < bb.y_max
            assert 0 < bb.confidence <= 1
