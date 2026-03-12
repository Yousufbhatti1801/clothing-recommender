"""
app/core/pinecone_client.py
============================
Thin shim that keeps the original public API (``init_pinecone`` /
``get_pinecone_index``) working while delegating to the unified
``PineconeVectorService`` singleton.

Both ``IngestionService`` (via FastAPI DI) and ``RecommendationPipeline``
now share a single Pinecone connection, so the index is created at most
once per process.
"""
from __future__ import annotations

from pinecone import Index


def init_pinecone() -> None:
    """Warm up the shared PineconeVectorService singleton.

    Called by the FastAPI lifespan hook on startup so the Pinecone index
    connection is established before the first request arrives.
    """
    from app.services.vector_store import get_vector_service
    get_vector_service()


def get_pinecone_index() -> Index:
    """Return the shared Pinecone ``Index`` used by all services.

    Delegates to the ``PineconeVectorService`` singleton so every caller
    (``IngestionService``, ``SearchService``, health checks) shares the
    same underlying connection and index object.
    """
    from app.services.vector_store import get_vector_service
    return get_vector_service()._index
