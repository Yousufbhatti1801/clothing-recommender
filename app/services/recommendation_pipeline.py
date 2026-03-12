"""RecommendationPipeline: image → YOLO → crop → CLIP → Pinecone → grouped results.

This service is intentionally decoupled from PostgreSQL.  It uses:
  • DetectAndEmbedPipeline  (YOLO detection + CLIP embedding)
  • PineconeVectorService   (index management + vector search)

and returns results grouped by garment category without needing a catalog DB.
"""
from __future__ import annotations

from PIL import Image

from app.models.schemas import (
    GarmentCategory,
    PipelineCategoryResult,
    PipelineMatch,
    PipelineRecommendationResponse,
)
from app.services.detect_and_embed import DetectAndEmbedPipeline, GarmentEmbedding
from app.services.vector_store import PineconeVectorService, get_vector_service


class RecommendationPipeline:
    """
    Six-step recommendation pipeline.

    Steps
    -----
    1. Accept a PIL Image (already loaded from the upload).
    2. Run YOLOv8 clothing detection → bounding boxes for shirts / pants / shoes.
    3. Crop each bounding box from the original image.
    4. Batch-encode all crops with CLIP → 512-d L2-normalised vectors.
    5. Query Pinecone per category (each category is a separate namespace).
    6. Return a PipelineRecommendationResponse grouped by clothing type.
    """

    def __init__(
        self,
        detect_embed: DetectAndEmbedPipeline | None = None,
        vector_store: PineconeVectorService | None = None,
        top_k: int = 5,
    ) -> None:
        self._detect_embed = detect_embed or DetectAndEmbedPipeline()
        self._vector_store = vector_store or get_vector_service()
        self._top_k = top_k

    async def run(
        self,
        image: Image.Image,
        budget: float | None = None,
    ) -> PipelineRecommendationResponse:
        """Execute the full pipeline and return grouped recommendations.

        Args:
            image:  PIL Image loaded from the uploaded file.
            budget: Optional maximum price per item (USD).  When provided,
                    a Pinecone server-side filter is applied so over-budget
                    products never appear in results.
        """
        # ── Steps 2-4: detect → crop → embed ────────────────────────────────
        pipeline_result = await self._detect_embed.run(image)

        if pipeline_result.total == 0:
            return PipelineRecommendationResponse(
                shirts=[], pants=[], shoes=[],
                total_detections=0,
                total_matches=0,
            )

        price_filter = {"price": {"$lte": budget}} if budget is not None else None

        # ── Steps 5-6: search Pinecone per category, group results ──────────
        shirts_results = self._search_category(
            pipeline_result.shirts, GarmentCategory.SHIRT, price_filter=price_filter
        )
        pants_results = self._search_category(
            pipeline_result.pants, GarmentCategory.PANTS, price_filter=price_filter
        )
        shoes_results = self._search_category(
            pipeline_result.shoes, GarmentCategory.SHOES, price_filter=price_filter
        )

        total_matches = (
            sum(len(r.matches) for r in shirts_results)
            + sum(len(r.matches) for r in pants_results)
            + sum(len(r.matches) for r in shoes_results)
        )

        return PipelineRecommendationResponse(
            shirts=shirts_results,
            pants=pants_results,
            shoes=shoes_results,
            total_detections=pipeline_result.total,
            total_matches=total_matches,
        )

    def _search_category(
        self,
        garment_embeddings: list[GarmentEmbedding],
        category: GarmentCategory,
        price_filter: dict | None = None,
    ) -> list[PipelineCategoryResult]:
        """
        For every detected crop in a category, query Pinecone and build a
        PipelineCategoryResult.  Multiple crops of the same category (e.g. two
        detected shirts) each produce their own result entry.
        """
        results: list[PipelineCategoryResult] = []
        for ge in garment_embeddings:
            vector_hits = self._vector_store.query(
                values=ge.embedding,
                namespace=category.value,
                top_k=self._top_k,
                filter=price_filter,
            )
            matches = [
                PipelineMatch(
                    product_id=hit.id,
                    score=round(hit.score, 4),
                    metadata=hit.metadata,
                    price=hit.metadata.get("price"),
                    brand=hit.metadata.get("brand") or None,
                    image_url=hit.metadata.get("image_url") or None,
                )
                for hit in vector_hits
            ]
            results.append(
                PipelineCategoryResult(
                    category=category,
                    detection_confidence=round(ge.confidence, 4),
                    matches=matches,
                )
            )
        return results
