"""Unit tests for SearchService — mocked Pinecone index."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.models.schemas import GarmentCategory, VectorMatch
from app.services.search import SearchService


@pytest.fixture
def search_service(mock_pinecone_index) -> SearchService:
    return SearchService(index=mock_pinecone_index)


@pytest.fixture
def random_embedding() -> np.ndarray:
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_vector_matches(self, search_service, random_embedding):
        results = await search_service.search(random_embedding, GarmentCategory.SHIRT)
        assert len(results) == 3
        for m in results:
            assert isinstance(m, VectorMatch)
            assert m.category == GarmentCategory.SHIRT

    @pytest.mark.asyncio
    async def test_respects_custom_top_k(self, search_service, random_embedding, mock_pinecone_index):
        await search_service.search(random_embedding, GarmentCategory.PANTS, top_k=10)
        call_kwargs = mock_pinecone_index.query.call_args.kwargs
        assert call_kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_uses_category_as_namespace(self, search_service, random_embedding, mock_pinecone_index):
        await search_service.search(random_embedding, GarmentCategory.SHOES)
        call_kwargs = mock_pinecone_index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "shoes"

    @pytest.mark.asyncio
    async def test_empty_matches(self, random_embedding):
        index = MagicMock()
        index.query = MagicMock(return_value={"matches": []})
        svc = SearchService(index=index)
        results = await svc.search(random_embedding, GarmentCategory.SHIRT)
        assert results == []


class TestSearchMany:
    @pytest.mark.asyncio
    async def test_merges_results(self, search_service, random_embedding):
        pairs = [
            (random_embedding, GarmentCategory.SHIRT),
            (random_embedding, GarmentCategory.PANTS),
        ]
        results = await search_service.search_many(pairs)
        # 3 matches per category × 2 categories
        assert len(results) == 6

    @pytest.mark.asyncio
    async def test_empty_pairs(self, search_service):
        results = await search_service.search_many([])
        assert results == []
