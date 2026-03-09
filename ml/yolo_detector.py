"""YOLOv8 clothing detector — singleton, loaded once at app startup."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.config import get_settings
from app.models.schemas import BoundingBox, DetectedGarment, GarmentCategory

# Maps YOLOv8 class indices → GarmentCategory.
# Update these indices when fine-tuning on DeepFashion2 or similar.
LABEL_MAP: dict[int, GarmentCategory] = {
    0: GarmentCategory.SHIRT,
    1: GarmentCategory.PANTS,
    2: GarmentCategory.SHOES,
    3: GarmentCategory.JACKET,
    4: GarmentCategory.DRESS,
    5: GarmentCategory.SKIRT,
}

# The three primary garment types this app focuses on.
TARGET_CATEGORIES: frozenset[GarmentCategory] = frozenset({
    GarmentCategory.SHIRT,
    GarmentCategory.PANTS,
    GarmentCategory.SHOES,
})


@dataclass
class _RawDetection:
    category: GarmentCategory
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class YOLODetector:
    """Wraps a fine-tuned YOLOv8 model for garment detection."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = YOLO(settings.yolo_model_path)
        self.confidence_threshold = settings.yolo_confidence_threshold

    def detect(self, image: Image.Image) -> list[DetectedGarment]:
        """
        Run inference on a single PIL Image.

        Returns:
            List of DetectedGarment, one per garment found above the
            confidence threshold.
        """
        img_array = np.array(image.convert("RGB"))
        results = self.model(img_array, conf=self.confidence_threshold, verbose=False)

        detections: list[DetectedGarment] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                category = LABEL_MAP.get(cls_idx, GarmentCategory.OTHER)
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())

                detections.append(
                    DetectedGarment(
                        category=category,
                        bounding_box=BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            confidence=confidence,
                        ),
                    )
                )

        return detections

    def detect_targets(
        self,
        image: Image.Image,
        categories: frozenset[GarmentCategory] = TARGET_CATEGORIES,
    ) -> list[DetectedGarment]:
        """
        Run detection and return only garments whose category is in *categories*.

        By default this returns shirt, pants, and shoes only — the three items
        needed for the recommendation pipeline.  Pass a custom frozenset to
        override at call-site.
        """
        return [g for g in self.detect(image) if g.category in categories]

    async def detect_async(self, image: Image.Image) -> list[DetectedGarment]:
        """Non-blocking wrapper — runs detect() in the default thread-pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect, image)

    async def detect_targets_async(
        self,
        image: Image.Image,
        categories: frozenset[GarmentCategory] = TARGET_CATEGORIES,
    ) -> list[DetectedGarment]:
        """Non-blocking version of detect_targets."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect_targets, image, categories)


@lru_cache(maxsize=1)
def get_yolo_detector() -> YOLODetector:
    """Return the process-wide singleton YOLODetector."""
    return YOLODetector()
