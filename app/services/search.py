"""SearchService: queries Pinecone for visually similar products."""
from __future__ import annotations

import numpy as np
from pinecone import Index

from app.core.config import get_settings
from app.models.schemas import GarmentCategory, VectorMatch


class SearchService:
    def __init__(self, index: Index) -> None:
        self._index = index
        self._settings = get_settings()

    async def search(
        self,
        embedding: np.ndarray,
        category: GarmentCategory,
        top_k: int | None = None,
    ) -> list[VectorMatch]:
        """
        Query Pinecone for the top-k most similar vectors in a
        category-specific namespace.

        Args:
            embedding: 512-d L2-normalised float32 vector.
            category:  Garment category (maps to Pinecone namespace).
            top_k:     Number of results (default from settings).

        Returns:
            List of VectorMatch sorted by descending cosine score.
        """
        k = top_k or self._settings.pinecone_top_k
        response = self._index.query(
            vector=embedding.tolist(),
            top_k=k,
            namespace=category.value,
            include_metadata=False,
        )
        return [
            VectorMatch(
                product_id=match["id"],
                score=float(match["score"]),
                category=category,
            )
            for match in response.get("matches", [])
        ]

    async def search_many(
        self,
        pairs: list[tuple[np.ndarray, GarmentCategory]],
        top_k: int | None = None,
    ) -> list[VectorMatch]:
        """Search for multiple (embedding, category) pairs and merge results."""
        all_matches: list[VectorMatch] = []
        for embedding, category in pairs:
            matches = await self.search(embedding, category, top_k)
            all_matches.extend(matches)
        return all_matches
