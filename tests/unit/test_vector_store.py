"""Unit tests for PineconeVectorService — mocked Pinecone client."""
from unittest.mock import MagicMock, patch

import pytest

from app.services.vector_store import PineconeVectorService


@pytest.fixture
def mock_pinecone_client():
    """Patches Pinecone at the module level so PineconeVectorService.__init__ works."""
    with patch("app.services.vector_store.Pinecone") as MockPinecone:
        client = MagicMock()
        client.list_indexes.return_value = [{"name": "clothing-embeddings"}]
        mock_index = MagicMock()
        client.Index.return_value = mock_index
        MockPinecone.return_value = client
        yield client, mock_index


class TestQuery:
    def test_returns_vector_results(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client
        mock_index.query.return_value = {
            "matches": [
                {"id": "p-1", "score": 0.95, "metadata": {"brand": "Nike"}},
                {"id": "p-2", "score": 0.88, "metadata": {"brand": "Adidas"}},
            ]
        }

        svc = PineconeVectorService()
        results = svc.query(values=[0.1] * 512, namespace="shirt", top_k=5)

        assert len(results) == 2
        assert results[0].id == "p-1"
        assert results[0].score == 0.95
        assert results[0].metadata == {"brand": "Nike"}

    def test_empty_matches(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client
        mock_index.query.return_value = {"matches": []}

        svc = PineconeVectorService()
        results = svc.query(values=[0.1] * 512, namespace="pants")
        assert results == []


class TestUpsert:
    def test_groups_by_namespace(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client

        svc = PineconeVectorService()
        svc.upsert([
            {"id": "v1", "values": [0.1] * 512, "metadata": {}, "namespace": "shirt"},
            {"id": "v2", "values": [0.2] * 512, "metadata": {}, "namespace": "shirt"},
            {"id": "v3", "values": [0.3] * 512, "metadata": {}, "namespace": "pants"},
        ])
        # Should have called index.upsert twice: once for "shirt", once for "pants"
        assert mock_index.upsert.call_count == 2

    def test_empty_vectors_no_op(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client

        svc = PineconeVectorService()
        svc.upsert([])
        mock_index.upsert.assert_not_called()


class TestQueryFilter:
    def test_filter_forwarded_to_pinecone(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client
        mock_index.query.return_value = {"matches": []}

        svc = PineconeVectorService()
        price_filter = {"price": {"$lte": 50.0}}
        svc.query(values=[0.1] * 512, namespace="shirt", top_k=5, filter=price_filter)

        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == price_filter

    def test_no_filter_kwarg_when_none(self, mock_pinecone_client):
        _, mock_index = mock_pinecone_client
        mock_index.query.return_value = {"matches": []}

        svc = PineconeVectorService()
        svc.query(values=[0.1] * 512, namespace="shirt", top_k=5)

        call_kwargs = mock_index.query.call_args.kwargs
        assert "filter" not in call_kwargs


class TestEnsureIndex:
    def test_does_not_recreate_existing_index(self, mock_pinecone_client):
        client, _ = mock_pinecone_client
        _ = PineconeVectorService()
        client.create_index.assert_not_called()

    def test_creates_index_when_missing(self):
        with patch("app.services.vector_store.Pinecone") as MockPinecone:
            client = MagicMock()
            client.list_indexes.return_value = []  # no existing indexes
            client.Index.return_value = MagicMock()
            MockPinecone.return_value = client

            _ = PineconeVectorService()
            client.create_index.assert_called_once()
