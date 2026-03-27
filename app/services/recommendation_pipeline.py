"""RecommendationPipeline: image → YOLO → crop → CLIP → Pinecone → grouped results.

This service is intentionally decoupled from PostgreSQL.  It uses:
  • DetectAndEmbedPipeline  (YOLO detection + CLIP embedding)
  • PineconeVectorService   (index management + vector search)

and returns results grouped by garment category without needing a catalog DB.
"""
from __future__ import annotations

import asyncio

from PIL import Image

from app.core.executors import get_ml_executor
from app.models.schemas import (
    GarmentCategory,
    PipelineCategoryResult,
    PipelineMatch,
    PipelineRecommendationResponse,
)
from app.services.detect_and_embed import DetectAndEmbedPipeline, GarmentEmbedding
from app.services.vector_store import PineconeVectorService, get_vector_service

# ── Quality thresholds ───────────────────────────────────────────────────────
# Minimum cosine similarity for a Pinecone match to be included in results.
# CLIP cosine scores typically range 0.55–0.95 for genuinely similar images.
# Anything below 0.60 is visually dissimilar and confuses users.
_MIN_MATCH_SCORE: float = 0.60


class RecommendationPipeline:
    """
    Six-step recommendation pipeline.

    Steps
    -----
    1. Accept a PIL Image (already loaded from the upload).
    2. Run YOLOv8 clothing detection → bounding boxes for all 6 fashion categories.
    3. Crop each bounding box from the original image.
    4. Batch-encode all crops with CLIP → 512-d L2-normalised vectors.
    5. Query Pinecone per category concurrently (each category is a separate namespace).
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
                jackets=[], dresses=[], skirts=[],
                total_detections=0,
                total_matches=0,
            )

        price_filter = {"price": {"$lte": budget}} if budget is not None else None

        # ── Steps 5-6: search all 6 Pinecone namespaces concurrently ────────
        (
            shirts_results,
            pants_results,
            shoes_results,
            jackets_results,
            dresses_results,
            skirts_results,
        ) = await asyncio.gather(
            self._search_category_async(
                pipeline_result.shirts, GarmentCategory.SHIRT, price_filter
            ),
            self._search_category_async(
                pipeline_result.pants, GarmentCategory.PANTS, price_filter
            ),
            self._search_category_async(
                pipeline_result.shoes, GarmentCategory.SHOES, price_filter
            ),
            self._search_category_async(
                pipeline_result.jackets, GarmentCategory.JACKET, price_filter
            ),
            self._search_category_async(
                pipeline_result.dresses, GarmentCategory.DRESS, price_filter
            ),
            self._search_category_async(
                pipeline_result.skirts, GarmentCategory.SKIRT, price_filter
            ),
        )

        total_matches = (
            sum(len(r.matches) for r in shirts_results)
            + sum(len(r.matches) for r in pants_results)
            + sum(len(r.matches) for r in shoes_results)
            + sum(len(r.matches) for r in jackets_results)
            + sum(len(r.matches) for r in dresses_results)
            + sum(len(r.matches) for r in skirts_results)
        )

        return PipelineRecommendationResponse(
            shirts=shirts_results,
            pants=pants_results,
            shoes=shoes_results,
            jackets=jackets_results,
            dresses=dresses_results,
            skirts=skirts_results,
            total_detections=pipeline_result.total,
            total_matches=total_matches,
        )

    async def _search_category_async(
        self,
        garment_embeddings: list[GarmentEmbedding],
        category: GarmentCategory,
        price_filter: dict | None = None,
    ) -> list[PipelineCategoryResult]:
        """Async wrapper — runs the synchronous Pinecone query off the event loop."""
        if not garment_embeddings:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_ml_executor(),
            lambda: self._search_category(garment_embeddings, category, price_filter),
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
                if hit.score >= _MIN_MATCH_SCORE  # drop visually dissimilar junk
            ]
            results.append(
                PipelineCategoryResult(
                    category=category,
                    detection_confidence=round(ge.confidence, 4),
                    matches=matches,
                )
            )
        return results
