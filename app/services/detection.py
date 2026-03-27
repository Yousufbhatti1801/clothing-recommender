"""DetectionService: crops garment regions from an uploaded image."""
from __future__ import annotations

from PIL import Image

from app.models.schemas import (
    BoundingBox,
    ClothingDetectionResponse,
    DetectedGarment,
)
from ml.yolo_detector import YOLODetector, get_yolo_detector

# Minimum CLIP zero-shot similarity to trust a classification result.
# Below this the image is probably ambiguous (e.g. a person with no garment).
_CLIP_MIN_CONFIDENCE: float = 0.20


class DetectionService:
    """
    Garment detection with automatic CLIP zero-shot fallback.

    Primary path:  YOLO detects bounding boxes → returns DetectedGarment list.
    Fallback path: when YOLO finds nothing, CLIP classifies the whole image as
                   a single garment category and returns a whole-image bbox.
                   This handles product flat-lay shots and other YOLO blind spots.
    """

    def __init__(
        self,
        detector: YOLODetector | None = None,
        *,
        clip_fallback: bool = True,
    ) -> None:
        self._detector = detector or get_yolo_detector()
        self._clip_fallback = clip_fallback

    async def detect(self, image: Image.Image) -> list[DetectedGarment]:
        """
        Return all garment detections.

        Uses YOLO as primary detector.  If YOLO finds nothing and
        ``clip_fallback=True``, falls back to CLIP zero-shot classification
        over the whole image, returning a single whole-image detection.
        """
        detections = await self._detector.detect_async(image)
        if not detections and self._clip_fallback:
            detections = await self._clip_fallback_detect(image)
        return detections

    async def _clip_fallback_detect(
        self, image: Image.Image
    ) -> list[DetectedGarment]:
        """
        CLIP zero-shot multi-region fallback.

        Splits the image into body zones (upper → shirt/jacket, middle → pants,
        lower → shoes) and classifies each region separately.  Returns one
        DetectedGarment per detected region with a bounding box spanning just
        that zone — NOT the full image.

        This produces far better Pinecone matches because each region's
        crop & embedding is focused on a single garment type instead of
        the entire outfit polluted into one vector.
        """
        from ml.clip_classifier import get_clip_classifier
        classifier = get_clip_classifier()

        region_results = await classifier.classify_regions_async(
            image, min_confidence=_CLIP_MIN_CONFIDENCE,
        )

        if not region_results:
            return []

        w, h = image.size
        detections: list[DetectedGarment] = []
        for category, confidence, _region_crop in region_results:
            # Find the y-bounds for the region this category came from
            y_min_px, y_max_px = 0.0, float(h)
            for y_start_r, y_end_r, allowed in classifier.BODY_REGIONS:
                if category in allowed:
                    y_min_px = h * y_start_r
                    y_max_px = h * y_end_r
                    break

            detections.append(
                DetectedGarment(
                    category=category,
                    bounding_box=BoundingBox(
                        x_min=0.0,
                        y_min=y_min_px,
                        x_max=float(w),
                        y_max=y_max_px,
                        confidence=confidence,
                    ),
                )
            )
        return detections

    def crop(self, image: Image.Image, garment: DetectedGarment) -> Image.Image:
        """Return the cropped sub-image for a single detected garment."""
        bb = garment.bounding_box
        return image.crop((bb.x_min, bb.y_min, bb.x_max, bb.y_max))

    async def detect_clothing(
        self, image: Image.Image
    ) -> ClothingDetectionResponse:
        """
        Detect only shirts, pants, and shoes from the image.

        Returns a ClothingDetectionResponse with results grouped per category
        and tiny bounding-box detections filtered out (< 1 % of image area).
        """
        raw = await self._detector.detect_all_fashion_async(image)
        img_area = image.width * image.height

        filtered: list[DetectedGarment] = []
        for garment in raw:
            bb = garment.bounding_box
            box_area = (bb.x_max - bb.x_min) * (bb.y_max - bb.y_min)
            if img_area > 0 and (box_area / img_area) < 0.01:
                continue  # discard likely false positives
            filtered.append(garment)

        return ClothingDetectionResponse.from_detections(filtered)

    async def detect_and_crop(
        self, image: Image.Image
    ) -> list[tuple[DetectedGarment, Image.Image]]:
        """
        Detect all garments and return (garment, crop) pairs.
        Filters out garments with bounding boxes smaller than 1 % of image area.
        """
        detections = await self.detect(image)
        img_area = image.width * image.height
        results = []
        for garment in detections:
            bb = garment.bounding_box
            box_area = (bb.x_max - bb.x_min) * (bb.y_max - bb.y_min)
            if img_area > 0 and (box_area / img_area) < 0.01:
                continue  # skip tiny false positives
            crop = self.crop(image, garment)
            results.append((garment, crop))
        return results
