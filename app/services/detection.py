"""DetectionService: crops garment regions from an uploaded image."""
from __future__ import annotations

from PIL import Image

from app.models.schemas import ClothingDetectionResponse, DetectedGarment, GarmentCategory
from ml.yolo_detector import TARGET_CATEGORIES, YOLODetector, get_yolo_detector


class DetectionService:
    def __init__(self, detector: YOLODetector | None = None) -> None:
        self._detector = detector or get_yolo_detector()

    async def detect(self, image: Image.Image) -> list[DetectedGarment]:
        """Return all garment detections for the given image."""
        return await self._detector.detect_async(image)

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
        raw = await self._detector.detect_targets_async(image)
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
