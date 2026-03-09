"""High-level Pinecone vector service: init, ensure index, upsert, query."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from pinecone import Index, Pinecone

from app.core.config import get_settings


@dataclass
class VectorResult:
    """Convenience wrapper for Pinecone query hits."""

    id: str
    score: float
    metadata: dict


class PineconeVectorService:
    """Encapsulates Pinecone client/index lifecycle plus CRUD helpers."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._ensure_index()
        self._index: Index = self._client.Index(settings.pinecone_index_name)

    def _ensure_index(self) -> None:
        """Create the index if it does not already exist."""
        index_name = self._settings.pinecone_index_name
        existing = {idx["name"] for idx in self._client.list_indexes()}
        if index_name in existing:
            return
        # Default to cosine metric since embeddings are L2-normalised.
        self._client.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
        )

    def upsert(self, vectors: Sequence[dict]) -> None:
        """
        Upsert embeddings.

        Each dict should contain: {"id": str, "values": list[float], "metadata": dict, "namespace": str}.
        Namespaces allow per-category partitioning (e.g., "shirt", "pants", "shoes").
        """
        if not vectors:
            return
        # Group by namespace so we minimise API calls.
        by_namespace: dict[str, list[dict]] = {}
        for vec in vectors:
            namespace = vec.get("namespace") or "default"
            payload = {k: v for k, v in vec.items() if k != "namespace"}
            by_namespace.setdefault(namespace, []).append(payload)
        for namespace, payloads in by_namespace.items():
            self._index.upsert(vectors=payloads, namespace=namespace)

    def query(
        self,
        values: Iterable[float],
        namespace: str,
        top_k: int = 5,
        with_metadata: bool = True,
    ) -> list[VectorResult]:
        """Return the top-k most similar embeddings from Pinecone."""
        response = self._index.query(
            vector=list(values),
            top_k=top_k,
            namespace=namespace,
            include_metadata=with_metadata,
        )
        matches = response.get("matches", [])
        results: list[VectorResult] = []
        for match in matches[:top_k]:
            results.append(
                VectorResult(
                    id=match.get("id"),
                    score=float(match.get("score", 0.0)),
                    metadata=match.get("metadata", {}) if with_metadata else {},
                )
            )
        return results
