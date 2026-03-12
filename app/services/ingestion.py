"""IngestionService: embeds a new catalog item and upserts it into Pinecone + PostgreSQL."""
from __future__ import annotations

import uuid

import httpx
from PIL import Image

from app.models.schemas import ProductIngestRequest, ProductIngestResponse
from app.services.catalog import CatalogService
from app.services.embedding import EmbeddingService
from pinecone import Index


class IngestionService:
    def __init__(
        self,
        catalog: CatalogService,
        embedding: EmbeddingService,
        index: Index,
    ) -> None:
        self._catalog = catalog
        self._embedding = embedding
        self._index = index

    async def ingest(self, data: ProductIngestRequest) -> ProductIngestResponse:
        """
        Full ingestion flow:
          1. Download product image
          2. Generate CLIP embedding
          3. Persist product row in PostgreSQL
          4. Upsert vector in Pinecone (namespace = category)
          5. Write vector_id back to the product row
        """
        # ── Download image ────────────────────────────────────────────────
        image = await self._download_image(str(data.image_url))

        # ── Embed ─────────────────────────────────────────────────────────
        vector = await self._embedding.embed_single(image)

        # ── Persist to PostgreSQL ─────────────────────────────────────────
        product = await self._catalog.create_product(data)
        vector_id = str(product.id)

        # ── Upsert to Pinecone (with metadata for server-side filtering) ─────
        metadata = {
            "price":     float(data.price),
            "currency":  data.currency,
            "brand":     data.brand or "",
            "category":  data.category.value,
            "name":      data.name,
            "seller_id": str(data.seller_id) if data.seller_id else "",
            "image_url": str(data.image_url),
        }
        self._index.upsert(
            vectors=[{"id": vector_id, "values": vector.tolist(), "metadata": metadata}],
            namespace=data.category.value,
        )

        # ── Write vector_id back ──────────────────────────────────────────
        await self._catalog.set_vector_id(product.id, vector_id)

        return ProductIngestResponse(product_id=product.id, vector_id=vector_id)

    @staticmethod
    async def _download_image(url: str) -> Image.Image:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        from io import BytesIO
        return Image.open(BytesIO(response.content)).convert("RGB")
