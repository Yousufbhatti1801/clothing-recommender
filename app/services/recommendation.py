"""RecommendationService: orchestrates the full AI pipeline."""
from __future__ import annotations

import logging

from PIL import Image

from app.core.config import get_settings
from app.models.schemas import (
    GarmentCategory,
    GarmentRecommendations,
    ProductResponse,
    RecommendationRequest,
    RecommendationResponse,
    SellerResponse,
)
from app.services.catalog import CatalogService
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.search import SearchService
from app.utils.geo import compute_locality_boost

log = logging.getLogger(__name__)


class RecommendationService:
    def __init__(
        self,
        detection: DetectionService,
        embedding: EmbeddingService,
        search: SearchService,
        catalog: CatalogService,
    ) -> None:
        self._detection = detection
        self._embedding = embedding
        self._search = search
        self._catalog = catalog
        self._settings = get_settings()

    async def recommend(
        self,
        image: Image.Image,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """
        Full pipeline:
          1. Detect garments
          2. Crop + embed each garment
          3. Search Pinecone per category (concurrently)
          4. Fetch product metadata from PostgreSQL
          5. Filter by budget, boost local sellers, rank
        """
        # ── Stage 1 & 2: detect + crop ────────────────────────────────────
        detected_pairs = await self._detection.detect_and_crop(image)
        if not detected_pairs:
            log.info("No garments detected in uploaded image.")
            return RecommendationResponse(
                results=[], detected_items=[], total_matches=0
            )

        # ── Stage 3: embed all crops ──────────────────────────────────────
        garments = [g for g, _ in detected_pairs]
        crops = [c for _, c in detected_pairs]
        embeddings = await self._embedding.embed(crops)

        # ── Stage 4: vector search (concurrent, per garment, pre-filtered) ─
        search_pairs = [(emb, g.category) for emb, g in zip(embeddings, garments, strict=False)]
        matches = await self._search.search_many(
            search_pairs, max_price=request.budget
        )
        log.info("Pinecone returned %d matches across %d garments.", len(matches), len(garments))

        # Group matches by category
        by_category: dict[GarmentCategory, list] = {}
        for match in matches:
            by_category.setdefault(match.category, []).append(match)

        # ── Stage 5: fetch catalog metadata ──────────────────────────────
        all_product_ids = [m.product_id for m in matches]
        products_map = await self._catalog.get_products_by_ids(all_product_ids)

        # ── Stage 6: filter + rank ────────────────────────────────────────
        results: list[GarmentRecommendations] = []
        total_matches = 0

        for category, cat_matches in by_category.items():
            ranked: list[ProductResponse] = []
            for match in cat_matches:
                product = products_map.get(match.product_id)
                if product is None:
                    continue

                locality_boost = 0.0
                is_local = False
                if (
                    request.user_latitude is not None
                    and request.user_longitude is not None
                    and product.seller
                    and product.seller.latitude is not None
                    and product.seller.longitude is not None
                ):
                    locality_boost = compute_locality_boost(
                        user_lat=request.user_latitude,
                        user_lon=request.user_longitude,
                        seller_lat=product.seller.latitude,
                        seller_lon=product.seller.longitude,
                        radius_km=self._settings.locality_radius_km,
                        boost=self._settings.locality_boost,
                    )
                    is_local = locality_boost > 0

                final_score = match.score + locality_boost

                seller_resp = None
                if product.seller:
                    seller_resp = SellerResponse.model_validate(product.seller)

                ranked.append(
                    ProductResponse(
                        id=product.id,
                        name=product.name,
                        brand=product.brand,
                        category=GarmentCategory(product.category),
                        price=product.price,
                        currency=product.currency,
                        image_url=product.image_url,
                        product_url=product.product_url,
                        seller=seller_resp,
                        similarity_score=round(final_score, 4),
                        is_local=is_local,
                    )
                )

            ranked = self._filter_by_budget(ranked, request.budget)
            ranked.sort(key=lambda p: p.similarity_score, reverse=True)
            top = ranked[: request.top_n]
            if top:
                results.append(GarmentRecommendations(category=category, items=top))
                total_matches += len(top)

        detected_categories = [g.category for g in garments]
        return RecommendationResponse(
            results=results,
            detected_items=detected_categories,
            total_matches=total_matches,
        )

    @staticmethod
    def _filter_by_budget(
        products: list[ProductResponse], budget: float | None
    ) -> list[ProductResponse]:
        """Return only products whose price is within the user's budget.
        If budget is None, all products are returned."""
        if budget is None:
            return products
        return [product for product in products if product.price <= budget]
