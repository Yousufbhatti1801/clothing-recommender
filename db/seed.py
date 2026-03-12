"""
db/seed.py
==========
Quick database seeder — inserts a small representative sample of sellers and
products with **real Unsplash image URLs** so the app is usable immediately
after ``docker compose up``.

This seeder writes to PostgreSQL only.  For a complete AI-ready catalog that
also upserts CLIP vectors + metadata to Pinecone, run the bulk seeder instead::

    python3 scripts/seed_catalog.py
"""
from __future__ import annotations

import asyncio
import uuid

from app.core.database import AsyncSessionLocal, engine
from app.models.orm import Base, Product, Seller

# ── Sellers ───────────────────────────────────────────────────────────────────
# Five geo-distributed sellers used across the product catalog.

SELLERS = [
    {
        "id": uuid.uuid4(),
        "name": "Urban Threads NYC",
        "city": "New York", "country": "US",
        "latitude": 40.7128, "longitude": -74.0060,
        "website": "https://urbanthreads.example.com",
    },
    {
        "id": uuid.uuid4(),
        "name": "Pacific Style LA",
        "city": "Los Angeles", "country": "US",
        "latitude": 34.0522, "longitude": -118.2437,
        "website": "https://pacificstyle.example.com",
    },
    {
        "id": uuid.uuid4(),
        "name": "Vintage Atelier London",
        "city": "London", "country": "GB",
        "latitude": 51.5074, "longitude": -0.1278,
        "website": "https://vintageatelier.example.com",
    },
    {
        "id": uuid.uuid4(),
        "name": "CoastLine Seattle",
        "city": "Seattle", "country": "US",
        "latitude": 47.6062, "longitude": -122.3321,
        "website": "https://coastline.example.com",
    },
    {
        "id": uuid.uuid4(),
        "name": "Heritage Shoe Boston",
        "city": "Boston", "country": "US",
        "latitude": 42.3601, "longitude": -71.0589,
        "website": "https://heritageshoe.example.com",
    },
]

# seller index (0-based) into the SELLERS list above
_S = {name: i for i, name in enumerate([s["name"] for s in SELLERS])}

# ── Products ──────────────────────────────────────────────────────────────────
# One representative product per category with a real, publicly accessible
# Unsplash image URL.  All seller_idx values are 0-based indices into SELLERS.

PRODUCTS = [
    # Shirts
    {"name": "Classic White Oxford Shirt",  "brand": "UrbanBasics",    "category": "shirt",  "price":  49.99, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=640"},
    {"name": "Navy Blue Linen Shirt",        "brand": "CoastLine",      "category": "shirt",  "price":  64.99, "currency": "USD", "seller_idx": 3, "image_url": "https://images.unsplash.com/photo-1598032895397-b9472444bf93?w=640"},
    {"name": "Grey Graphic Hoodie",          "brand": "StreetLabel",    "category": "shirt",  "price":  79.00, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=640"},
    # Pants
    {"name": "Slim-Fit Blue Jeans",          "brand": "DenimWorks",     "category": "pants",  "price":  89.99, "currency": "USD", "seller_idx": 1, "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=640"},
    {"name": "Khaki Chino Pants",            "brand": "UrbanBasics",    "category": "pants",  "price":  69.99, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=640"},
    {"name": "Olive Linen Trousers",         "brand": "CoastLine",      "category": "pants",  "price":  86.00, "currency": "USD", "seller_idx": 3, "image_url": "https://images.unsplash.com/photo-1604176354204-9268737828e4?w=640"},
    # Shoes
    {"name": "White Canvas Sneakers",        "brand": "PacificKicks",   "category": "shoes",  "price":  79.99, "currency": "USD", "seller_idx": 1, "image_url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=640"},
    {"name": "Black Leather Oxfords",        "brand": "HeritageShoe",   "category": "shoes",  "price": 185.00, "currency": "USD", "seller_idx": 4, "image_url": "https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=640"},
    {"name": "Tan Suede Chelsea Boots",      "brand": "HeritageShoe",   "category": "shoes",  "price": 219.00, "currency": "USD", "seller_idx": 4, "image_url": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=640"},
    # Jackets
    {"name": "Leather Biker Jacket",         "brand": "UrbanBasics",    "category": "jacket", "price": 199.99, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=640"},
    {"name": "Caramel Trench Coat",          "brand": "VintageAtelier", "category": "jacket", "price": 320.00, "currency": "USD", "seller_idx": 2, "image_url": "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=640"},
    {"name": "Navy Wool Blazer",             "brand": "OfficePro",      "category": "jacket", "price": 245.00, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=640"},
    # Dresses
    {"name": "Floral Midi Dress",            "brand": "SummerDays",     "category": "dress",  "price": 119.00, "currency": "USD", "seller_idx": 1, "image_url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=640"},
    {"name": "White Linen Sun Dress",        "brand": "SummerDays",     "category": "dress",  "price":  95.00, "currency": "USD", "seller_idx": 1, "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=640"},
    {"name": "Emerald Wrap Dress",           "brand": "VintageAtelier", "category": "dress",  "price": 155.00, "currency": "USD", "seller_idx": 2, "image_url": "https://images.unsplash.com/photo-1618932260643-eee4a2f652a6?w=640"},
    # Skirts
    {"name": "Pleated Midi Skirt",           "brand": "OfficePro",      "category": "skirt",  "price":  79.99, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1592301933927-35b597393c0a?w=640"},
    {"name": "Black Faux-Leather Skirt",     "brand": "UrbanBasics",    "category": "skirt",  "price": 119.00, "currency": "USD", "seller_idx": 0, "image_url": "https://images.unsplash.com/photo-1571513722275-4b41940f54b8?w=640"},
    {"name": "Tartan Plaid Kilt",            "brand": "VintageAtelier", "category": "skirt",  "price":  89.00, "currency": "USD", "seller_idx": 2, "image_url": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=640"},
]


async def seed() -> None:
    """Create tables (idempotent) then insert sellers + products."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # ── Sellers ───────────────────────────────────────────────────────
        seller_objs: list[Seller] = []
        for s in SELLERS:
            data = {k: v for k, v in s.items() if k != "website"}
            obj = Seller(**data)
            # store website if the ORM column exists
            if hasattr(obj, "website"):
                obj.website = s.get("website")
            session.add(obj)
            seller_objs.append(obj)
        await session.flush()  # populate obj.id

        # ── Products ──────────────────────────────────────────────────────
        for p in PRODUCTS:
            seller = seller_objs[p["seller_idx"]]
            product = Product(
                name=p["name"],
                brand=p["brand"],
                category=p["category"],
                price=p["price"],
                currency=p["currency"],
                image_url=p["image_url"],
                seller_id=seller.id,
            )
            session.add(product)

        await session.commit()

    print(f"✅  Seeded {len(seller_objs)} sellers and {len(PRODUCTS)} products.")
    print()
    print("  ℹ️  This seeder writes to PostgreSQL only.")
    print("     To also populate the Pinecone vector index, run:")
    print("       python3 scripts/seed_catalog.py")


if __name__ == "__main__":
    asyncio.run(seed())
