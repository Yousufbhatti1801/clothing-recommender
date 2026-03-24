"""DetectAndEmbedPipeline: YOLO detection → crop → CLIP embedding in one call."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.models.schemas import DetectedGarment, GarmentCategory
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from ml.clip_encoder import get_clip_encoder
from ml.yolo_detector import get_yolo_detector


@dataclass
class GarmentEmbedding:
    """The detection metadata and its CLIP vector for a single garment crop."""

    garment: DetectedGarment
    crop: Image.Image
    embedding: np.ndarray  # shape (512,), L2-normalised float32

    @property
    def category(self) -> GarmentCategory:
        return self.garment.category

    @property
    def confidence(self) -> float:
        return self.garment.bounding_box.confidence


@dataclass
class PipelineResult:
    """All garment embeddings grouped by category for a single image."""

    shirts: list[GarmentEmbedding] = field(default_factory=list)
    pants: list[GarmentEmbedding] = field(default_factory=list)
    shoes: list[GarmentEmbedding] = field(default_factory=list)
    jackets: list[GarmentEmbedding] = field(default_factory=list)
    dresses: list[GarmentEmbedding] = field(default_factory=list)
    skirts: list[GarmentEmbedding] = field(default_factory=list)

    @property
    def all(self) -> list[GarmentEmbedding]:
        """Flat list of all detected garment embeddings."""
        return (
            self.shirts + self.pants + self.shoes
            + self.jackets + self.dresses + self.skirts
        )

    @property
    def total(self) -> int:
        return (
            len(self.shirts) + len(self.pants) + len(self.shoes)
            + len(self.jackets) + len(self.dresses) + len(self.skirts)
        )


class DetectAndEmbedPipeline:
    """
    Single-call pipeline: image → detect clothing → crop → CLIP embeddings.

    Steps
    -----
    1. Run YOLOv8 on the image — returns bounding boxes for shirts, pants, shoes.
    2. Crop each bounding box from the original image.
    3. Batch-encode all crops with CLIP in a single forward pass.
    4. Return a PipelineResult grouped by category.

    Usage
    -----
    pipeline = DetectAndEmbedPipeline()
    result   = await pipeline.run(pil_image)

    for ge in result.shirts:
        print(ge.category, ge.embedding.shape, ge.confidence)
    """

    def __init__(
        self,
        detection_service: DetectionService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._detector  = detection_service  or DetectionService(get_yolo_detector())
        self._embedder  = embedding_service  or EmbeddingService(get_clip_encoder())

    async def run(self, image: Image.Image) -> PipelineResult:
        """
        Execute the full detect → crop → embed pipeline.

        Args:
            image: A PIL Image (any mode; will be normalised internally).

        Returns:
            PipelineResult with one GarmentEmbedding per detected garment,
            grouped by category.
        """
        # ── Step 1: detect shirts / pants / shoes ───────────────────────────
        garment_crop_pairs = await self._detector.detect_and_crop(image)

        if not garment_crop_pairs:
            return PipelineResult()

        # ── Step 2: separate garments and crops ─────────────────────────────
        garments = [g for g, _ in garment_crop_pairs]
        crops    = [c for _, c in garment_crop_pairs]

        # ── Step 3: batch-embed all crops in one CLIP forward pass ──────────
        vectors = await self._embedder.embed(crops)

        # ── Step 4: assemble into GarmentEmbedding and group by category ────
        result = PipelineResult()
        for garment, crop, vector in zip(garments, crops, vectors, strict=False):
            ge = GarmentEmbedding(garment=garment, crop=crop, embedding=vector)
            if garment.category == GarmentCategory.SHIRT:
                result.shirts.append(ge)
            elif garment.category == GarmentCategory.PANTS:
                result.pants.append(ge)
            elif garment.category == GarmentCategory.SHOES:
                result.shoes.append(ge)
            elif garment.category == GarmentCategory.JACKET:
                result.jackets.append(ge)
            elif garment.category == GarmentCategory.DRESS:
                result.dresses.append(ge)
            elif garment.category == GarmentCategory.SKIRT:
                result.skirts.append(ge)
            # GarmentCategory.OTHER is intentionally ignored

        return result
