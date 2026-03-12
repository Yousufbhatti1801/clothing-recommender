"""
tests/unit/test_search_filtering.py
=====================================
Verify SearchService budget pre-filtering behaviour:
  • ``max_price`` is translated to a Pinecone server-side ``filter`` dict.
  • No filter dict is sent when ``max_price`` is None.
  • ``include_metadata=True`` is always forwarded.
  • ``search_many`` threads ``max_price`` to every per-category call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.models.schemas import GarmentCategory, VectorMatch
from app.services.search import SearchService


@pytest.fixture
def embedding() -> np.ndarray:
    rng = np.random.default_rng(99)
    v = rng.random(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_index(matches: list[dict] | None = None) -> MagicMock:
    """Return a mock Pinecone index that returns *matches* for any query."""
    if matches is None:
        matches = [
            {"id": "p-1", "score": 0.9, "metadata": {"price": 40.0}},
            {"id": "p-2", "score": 0.8, "metadata": {"price": 120.0}},
        ]
    idx = MagicMock()
    idx.query = MagicMock(return_value={"matches": matches})
    return idx


# ── Budget filter ─────────────────────────────────────────────────────────────

class TestBudgetFilter:
    @pytest.mark.asyncio
    async def test_filter_sent_when_max_price_given(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        await svc.search(embedding, GarmentCategory.SHIRT, max_price=100.0)

        kwargs = idx.query.call_args.kwargs
        assert "filter" in kwargs
        assert kwargs["filter"] == {"price": {"$lte": 100.0}}

    @pytest.mark.asyncio
    async def test_no_filter_when_max_price_is_none(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        await svc.search(embedding, GarmentCategory.SHIRT, max_price=None)

        kwargs = idx.query.call_args.kwargs
        assert "filter" not in kwargs

    @pytest.mark.asyncio
    async def test_no_filter_when_max_price_omitted(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        await svc.search(embedding, GarmentCategory.SHIRT)

        kwargs = idx.query.call_args.kwargs
        assert "filter" not in kwargs

    @pytest.mark.asyncio
    async def test_filter_uses_lte_operator(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        budget = 49.99
        await svc.search(embedding, GarmentCategory.PANTS, max_price=budget)

        sent_filter = idx.query.call_args.kwargs["filter"]
        assert sent_filter["price"]["$lte"] == pytest.approx(budget)

    @pytest.mark.asyncio
    async def test_search_many_forwards_max_price(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        pairs = [
            (embedding, GarmentCategory.SHIRT),
            (embedding, GarmentCategory.PANTS),
        ]
        await svc.search_many(pairs, max_price=75.0)

        # Both calls must carry the filter
        for call in idx.query.call_args_list:
            assert call.kwargs.get("filter") == {"price": {"$lte": 75.0}}

    @pytest.mark.asyncio
    async def test_search_many_no_filter_when_no_budget(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        pairs = [(embedding, GarmentCategory.SHIRT)]
        await svc.search_many(pairs)

        kwargs = idx.query.call_args.kwargs
        assert "filter" not in kwargs


# ── include_metadata always True ──────────────────────────────────────────────

class TestIncludeMetadata:
    @pytest.mark.asyncio
    async def test_include_metadata_true_without_budget(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        await svc.search(embedding, GarmentCategory.SHOES)

        assert idx.query.call_args.kwargs["include_metadata"] is True

    @pytest.mark.asyncio
    async def test_include_metadata_true_with_budget(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        await svc.search(embedding, GarmentCategory.SHOES, max_price=200.0)

        assert idx.query.call_args.kwargs["include_metadata"] is True

    @pytest.mark.asyncio
    async def test_returns_vector_matches(self, embedding):
        idx = _make_index()
        svc = SearchService(index=idx)

        results = await svc.search(embedding, GarmentCategory.SHIRT)

        assert len(results) == 2
        assert all(isinstance(r, VectorMatch) for r in results)
