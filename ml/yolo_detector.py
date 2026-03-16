"""YOLOv8 clothing detector — singleton, loaded once at app startup."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.config import get_settings
from app.core.executors import get_ml_executor
from app.models.schemas import BoundingBox, DetectedGarment, GarmentCategory
from ml.fashion_classes import build_label_map_from_model_names
from ml.fashion_classes import is_fashion_model as _is_fashion_model

log = logging.getLogger(__name__)

# ── Backward-compatible static LABEL_MAP (6-class legacy schema) ─────────────
# This map is used as fallback when the model does not carry class names
# that match any known fashion vocabulary (e.g. a vanilla COCO model).
# Once yolov8_fashion.pt is fine-tuned and deployed, the detector uses its
# own dynamic map built from model.names automatically.
LABEL_MAP: dict[int, GarmentCategory] = {
    0: GarmentCategory.SHIRT,
    1: GarmentCategory.PANTS,
    2: GarmentCategory.SHOES,
    3: GarmentCategory.JACKET,
    4: GarmentCategory.DRESS,
    5: GarmentCategory.SKIRT,
}

# ── Target categories for the recommendation pipeline ────────────────────────
# The three garment types the pipeline focuses on (used by detect_targets).
TARGET_CATEGORIES: frozenset[GarmentCategory] = frozenset({
    GarmentCategory.SHIRT,
    GarmentCategory.PANTS,
    GarmentCategory.SHOES,
})

# Extended set — includes jacket, dress, skirt for broader detection
ALL_FASHION_CATEGORIES: frozenset[GarmentCategory] = frozenset({
    GarmentCategory.SHIRT,
    GarmentCategory.PANTS,
    GarmentCategory.SHOES,
    GarmentCategory.JACKET,
    GarmentCategory.DRESS,
    GarmentCategory.SKIRT,
})

# ── Detection quality filter ──────────────────────────────────────────────────
# Discard any detection whose bounding box covers less than this fraction of
# the total image area.  Tiny boxes (< 0.5 % of the image) are almost always
# false positives on accessories, text, or background textures.
MIN_BOX_AREA_RATIO: float = 0.005


@dataclass
class _RawDetection:
    category: GarmentCategory
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class YOLODetector:
    """
    Wraps a YOLOv8 model for garment detection.

    Supports two modes automatically:
    ─────────────────────────────────
    Fashion model (yolov8_fashion.pt after fine-tuning)
        model.names contains clothing terms such as "shirt", "pants", "shoes".
        The label map is built dynamically from model.names via
        ``ml.fashion_classes.build_label_map_from_model_names``, so any
        fine-tuned model works without code changes.

    COCO base model (yolov8n.pt / yolov8s.pt etc.)
        model.names contains COCO-80 terms ("person", "car", …).
        Most map to GarmentCategory.OTHER; a handful of accessory classes
        (handbag, backpack, tie) also map to OTHER.
        In this mode ``is_fashion_model`` is False and a diagnostics warning
        is issued so the operator knows to run the fine-tuning pipeline.

    After running ``scripts/train_fashion_yolo.py``, replace
    ``ml/models/yolov8_fashion.pt`` with the trained weights.  The next
    application restart will pick up the new model automatically.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model = YOLO(settings.yolo_model_path)
        self.confidence_threshold = settings.yolo_confidence_threshold

        # ── Build label map dynamically from the loaded model's class names ──
        # This works for any model — COCO, DeepFashion2, Fashionpedia, or the
        # app-native 13-class fine-tuned model.
        self._label_map: dict[int, GarmentCategory] = \
            build_label_map_from_model_names(self.model.names)

        # ── Diagnostics ──────────────────────────────────────────────────────
        self.is_fashion_model: bool = _is_fashion_model(self.model.names)
        self.model_class_names: dict[int, str] = dict(self.model.names)

        if not self.is_fashion_model:
            import warnings
            warnings.warn(
                f"[YOLODetector] Loaded model has {len(self.model.names)} COCO-style "
                "classes and is NOT fashion-specific. "
                "Garment detection will be very limited until you fine-tune the model. "
                "Run: python scripts/train_fashion_yolo.py",
                stacklevel=2,
            )

    # ── Core inference ────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> list[DetectedGarment]:
        """
        Run inference on a single PIL Image.

        Uses the dynamic label map built from the model's own class names,
        so the same code works for any fine-tuned model without changes.

        Returns:
            List of DetectedGarment, one per garment found above the
            confidence threshold, ordered by descending confidence.
        """
        img_array = np.array(image.convert("RGB"))
        results = self.model(img_array, conf=self.confidence_threshold, verbose=False)

        detections: list[DetectedGarment] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_idx    = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                # Discard tiny false-positive boxes (< MIN_BOX_AREA_RATIO of image)
                img_w, img_h = image.size
                box_area = (x_max - x_min) * (y_max - y_min)
                img_area = img_w * img_h
                if img_area > 0 and (box_area / img_area) < MIN_BOX_AREA_RATIO:
                    continue

                # Use dynamic map; fall back to OTHER for unknown indices
                category = self._label_map.get(cls_idx, GarmentCategory.OTHER)

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

        # Sort by descending confidence so the highest-confidence garments come first
        detections.sort(key=lambda d: d.bounding_box.confidence, reverse=True)
        return detections

    def detect_targets(
        self,
        image: Image.Image,
        categories: frozenset[GarmentCategory] = TARGET_CATEGORIES,
    ) -> list[DetectedGarment]:
        """
        Run detection and return only garments whose category is in *categories*.

        By default returns shirt, pants, and shoes only — the three items
        needed for the recommendation pipeline.  Pass ``ALL_FASHION_CATEGORIES``
        to include jacket, dress, and skirt as well.
        """
        return [g for g in self.detect(image) if g.category in categories]

    def detect_all_fashion(self, image: Image.Image) -> list[DetectedGarment]:
        """
        Convenience wrapper: detect all six primary fashion categories
        (shirt, pants, shoes, jacket, dress, skirt).
        """
        return self.detect_targets(image, ALL_FASHION_CATEGORIES)

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def detect_async(self, image: Image.Image) -> list[DetectedGarment]:
        """Non-blocking wrapper — runs detect() in the dedicated ML thread-pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_ml_executor(), self.detect, image)

    async def detect_targets_async(
        self,
        image: Image.Image,
        categories: frozenset[GarmentCategory] = TARGET_CATEGORIES,
    ) -> list[DetectedGarment]:
        """Non-blocking version of detect_targets."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_ml_executor(), self.detect_targets, image, categories)

    async def detect_all_fashion_async(
        self, image: Image.Image
    ) -> list[DetectedGarment]:
        """Non-blocking version of detect_all_fashion."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_ml_executor(), self.detect_all_fashion, image)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def model_summary(self) -> dict:
        """
        Return a summary dict useful for health checks and logging.

        Example return value::

            {
                "model_path": "ml/models/yolov8_fashion.pt",
                "is_fashion_model": True,
                "num_classes": 13,
                "class_names": {0: "shirt", 1: "pants", ...},
                "confidence_threshold": 0.4,
            }
        """
        return {
            "model_path": str(get_settings().yolo_model_path),
            "is_fashion_model": self.is_fashion_model,
            "num_classes": len(self.model_class_names),
            "class_names": self.model_class_names,
            "confidence_threshold": self.confidence_threshold,
        }

    def annotate_image(
        self,
        image: Image.Image,
        detections: list[DetectedGarment],
    ) -> Image.Image:
        """
        Draw bounding boxes and category labels on *image*.

        Returns a new PIL Image (the original is not mutated).  Useful for
        debugging — dump the result to disk with ``img.save('debug.jpg')``.

        Example::

            detector = get_yolo_detector()
            img      = Image.open("photo.jpg")
            dets     = detector.detect(img)
            debug    = detector.annotate_image(img, dets)
            debug.save("debug_annotated.jpg")
        """
        from PIL import ImageDraw, ImageFont

        # palette: one colour per GarmentCategory value
        _COLOURS: dict[GarmentCategory, str] = {
            GarmentCategory.SHIRT:   "#FF6B6B",
            GarmentCategory.PANTS:   "#4ECDC4",
            GarmentCategory.SHOES:   "#45B7D1",
            GarmentCategory.JACKET:  "#96CEB4",
            GarmentCategory.DRESS:   "#FFEAA7",
            GarmentCategory.SKIRT:   "#DDA0DD",
            GarmentCategory.OTHER:   "#B0B0B0",
        }

        annotated = image.copy().convert("RGB")
        draw      = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=16)
        except OSError:
            font = ImageFont.load_default()

        for det in detections:
            bb     = det.bounding_box
            colour = _COLOURS.get(det.category, "#B0B0B0")
            box    = [bb.x_min, bb.y_min, bb.x_max, bb.y_max]

            # bounding box (3-px border)
            draw.rectangle(box, outline=colour, width=3)

            # label background + text
            label = f"{det.category.value}  {bb.confidence:.0%}"
            try:
                text_bbox = draw.textbbox((bb.x_min, bb.y_min - 20), label, font=font)
                draw.rectangle(text_bbox, fill=colour)
            except AttributeError:
                # Pillow < 9.2 does not have textbbox
                pass
            draw.text(
                (bb.x_min + 2, bb.y_min - 20),
                label,
                fill="black",
                font=font,
            )

        return annotated


@lru_cache(maxsize=1)
def get_yolo_detector() -> YOLODetector:
    """Return the process-wide singleton YOLODetector."""
    return YOLODetector()
