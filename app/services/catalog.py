"""CatalogService: fetches and stores product + seller data in PostgreSQL."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.orm import Product, Seller
from app.models.schemas import ProductIngestRequest


class CatalogService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_products_by_ids(
        self, product_ids: list[str]
    ) -> dict[str, Product]:
        """
        Batch-fetch products by their UUID strings.
        Returns a dict keyed by str(product.id).
        """
        if not product_ids:
            return {}
        uuids = [uuid.UUID(pid) for pid in product_ids]
        stmt = (
            select(Product)
            .where(Product.id.in_(uuids))
            .options(selectinload(Product.seller))
        )
        result = await self._db.execute(stmt)
        products = result.scalars().all()
        return {str(p.id): p for p in products}

    async def create_product(self, data: ProductIngestRequest) -> Product:
        """Persist a new product row (without a vector_id yet)."""
        product = Product(
            name=data.name,
            brand=data.brand,
            description=data.description,
            category=data.category.value,
            price=data.price,
            currency=data.currency,
            image_url=str(data.image_url),
            product_url=str(data.product_url) if data.product_url else None,
            seller_id=data.seller_id,
        )
        self._db.add(product)
        await self._db.flush()  # populate product.id without committing
        return product

    async def set_vector_id(self, product_id: uuid.UUID, vector_id: str) -> None:
        """Update the Pinecone vector_id for an existing product."""
        stmt = select(Product).where(Product.id == product_id)
        result = await self._db.execute(stmt)
        product = result.scalar_one()
        product.vector_id = vector_id

    async def get_seller(self, seller_id: uuid.UUID) -> Seller | None:
        stmt = select(Seller).where(Seller.id == seller_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()
