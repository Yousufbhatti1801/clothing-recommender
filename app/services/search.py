"""SearchService: queries Pinecone for visually similar products."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
from pinecone import Index

from app.core.config import get_settings
from app.core.executors import get_ml_executor
from app.models.schemas import GarmentCategory, VectorMatch

log = logging.getLogger(__name__)


class SearchService:
    def __init__(self, index: Index) -> None:
        self._index = index
        self._settings = get_settings()

    def _search_sync(
        self,
        embedding: np.ndarray,
        category: GarmentCategory,
        top_k: int,
        max_price: float | None,
    ) -> list[VectorMatch]:
        """Synchronous Pinecone query — runs inside a thread-pool executor."""
        query_kwargs: dict = dict(
            vector=embedding.tolist(),
            top_k=top_k,
            namespace=category.value,
            include_metadata=True,
        )
        if max_price is not None:
            query_kwargs["filter"] = {"price": {"$lte": max_price}}
        response = self._index.query(**query_kwargs)
        return [
            VectorMatch(
                product_id=match["id"],
                score=float(match["score"]),
                category=category,
            )
            for match in response.get("matches", [])
        ]

    async def search(
        self,
        embedding: np.ndarray,
        category: GarmentCategory,
        top_k: int | None = None,
        max_price: float | None = None,
    ) -> list[VectorMatch]:
        """
        Query Pinecone for the top-k most similar vectors in a
        category-specific namespace. Runs the synchronous Pinecone SDK call
        in the dedicated ML thread-pool executor.

        Args:
            embedding:  512-d L2-normalised float32 vector.
            category:   Garment category (maps to Pinecone namespace).
            top_k:      Number of results (default from settings).
            max_price:  When set, adds a Pinecone server-side filter.

        Returns:
            List of VectorMatch sorted by descending cosine score.
        """
        k = top_k or self._settings.pinecone_top_k
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_ml_executor(),
            lambda: self._search_sync(embedding, category, k, max_price),
        )

    async def search_many(
        self,
        pairs: list[tuple[np.ndarray, GarmentCategory]],
        top_k: int | None = None,
        max_price: float | None = None,
    ) -> list[VectorMatch]:
        """
        Search for multiple (embedding, category) pairs **concurrently**
        using asyncio.gather — all Pinecone queries fire in parallel.
        """
        if not pairs:
            return []
        tasks = [self.search(emb, cat, top_k, max_price) for emb, cat in pairs]
        results = await asyncio.gather(*tasks)
        return [match for matches in results for match in matches]
