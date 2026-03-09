"""Seed the database with sample sellers and products."""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import insert

from app.core.database import AsyncSessionLocal, engine
from app.models.orm import Base, Product, Seller

SELLERS = [
    {"id": uuid.uuid4(), "name": "Urban Threads NYC", "city": "New York", "country": "US", "latitude": 40.7128, "longitude": -74.0060},
    {"id": uuid.uuid4(), "name": "Pacific Style LA", "city": "Los Angeles", "country": "US", "latitude": 34.0522, "longitude": -118.2437},
    {"id": uuid.uuid4(), "name": "Vintage Atelier London", "city": "London", "country": "GB", "latitude": 51.5074, "longitude": -0.1278},
]

PRODUCTS = [
    {"name": "Classic White Oxford Shirt", "brand": "UrbanBasics", "category": "shirt", "price": 49.99, "currency": "USD", "image_url": "https://example.com/img/white-oxford.jpg"},
    {"name": "Slim Fit Chinos", "brand": "UrbanBasics", "category": "pants", "price": 69.99, "currency": "USD", "image_url": "https://example.com/img/chinos.jpg"},
    {"name": "White Leather Sneakers", "brand": "PacificKicks", "category": "shoes", "price": 89.99, "currency": "USD", "image_url": "https://example.com/img/sneakers.jpg"},
    {"name": "Floral Summer Dress", "brand": "VintageAtelier", "category": "dress", "price": 119.00, "currency": "USD", "image_url": "https://example.com/img/floral-dress.jpg"},
    {"name": "Leather Biker Jacket", "brand": "UrbanBasics", "category": "jacket", "price": 199.99, "currency": "USD", "image_url": "https://example.com/img/biker-jacket.jpg"},
]


async def seed() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        seller_objs = []
        for s in SELLERS:
            obj = Seller(**s)
            session.add(obj)
            seller_objs.append(obj)
        await session.flush()

        for i, p in enumerate(PRODUCTS):
            product = Product(seller_id=seller_objs[i % len(seller_objs)].id, **p)
            session.add(product)

        await session.commit()
        print(f"✅  Seeded {len(SELLERS)} sellers and {len(PRODUCTS)} products.")


if __name__ == "__main__":
    asyncio.run(seed())
