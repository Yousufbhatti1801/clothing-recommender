#!/usr/bin/env python3
"""
scripts/seed_catalog.py
========================
Bulk-seed the catalog: download fashion product images, CLIP-embed them,
write to PostgreSQL, and upsert vectors to Pinecone with full metadata.

After running this script, ``POST /api/v1/pipeline/recommend`` will return
real recommendations from the seeded catalog.

Usage
-----
    # Seed with defaults (requires .env with PINECONE_API_KEY + running DB)
    python3.11 scripts/seed_catalog.py

    # Reset: drop all existing products/sellers first, then re-seed
    python3.11 scripts/seed_catalog.py --reset

    # Dry run: validate images are reachable without writing anything
    python3.11 scripts/seed_catalog.py --dry-run

    # Override embedding batch size (default 8)
    python3.11 scripts/seed_catalog.py --batch 16

Prerequisites
-------------
    • PINECONE_API_KEY set in .env or environment
    • PostgreSQL running and accessible via DB_URL (.env)
    • pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Catalog data ───────────────────────────────────────────────────────────────
# seller index (0-based) matches SELLERS list below

SELLERS = [
    {
        "name": "Urban Threads NYC",
        "city": "New York", "country": "US",
        "latitude": 40.7128, "longitude": -74.0060,
        "website": "https://urbanthreads.example.com",
    },
    {
        "name": "Pacific Style LA",
        "city": "Los Angeles", "country": "US",
        "latitude": 34.0522, "longitude": -118.2437,
        "website": "https://pacificstyle.example.com",
    },
    {
        "name": "Vintage Atelier London",
        "city": "London", "country": "GB",
        "latitude": 51.5074, "longitude": -0.1278,
        "website": "https://vintageatelier.example.com",
    },
    {
        "name": "CoastLine Seattle",
        "city": "Seattle", "country": "US",
        "latitude": 47.6062, "longitude": -122.3321,
        "website": "https://coastline.example.com",
    },
    {
        "name": "Heritage Shoe Boston",
        "city": "Boston", "country": "US",
        "latitude": 42.3601, "longitude": -71.0589,
        "website": "https://heritageshoe.example.com",
    },
]

# fmt: off
CATALOG: list[dict] = [
    # ── Shirts ────────────────────────────────────────────────────────────────
    {"name": "Classic White Oxford Shirt", "brand": "UrbanBasics",    "category": "shirt", "price":  49.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=640"},
    {"name": "Navy Blue Linen Shirt",      "brand": "CoastLine",      "category": "shirt", "price":  64.99, "seller": 3, "image_url": "https://images.unsplash.com/photo-1598032895397-b9472444bf93?w=640"},
    {"name": "Striped Casual Tee",         "brand": "UrbanBasics",    "category": "shirt", "price":  24.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=640"},
    {"name": "Black Slim-Fit Polo",        "brand": "SportEdge",      "category": "shirt", "price":  39.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=640"},
    {"name": "Plaid Flannel Shirt",        "brand": "OutdoorCo",      "category": "shirt", "price":  55.00, "seller": 3, "image_url": "https://images.unsplash.com/photo-1607522370275-f6fefe5f2bfb?w=640"},
    {"name": "Grey Graphic Hoodie",        "brand": "StreetLabel",    "category": "shirt", "price":  79.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=640"},
    {"name": "White V-Neck Tee",           "brand": "UrbanBasics",    "category": "shirt", "price":  19.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1554568218-0f1715e72254?w=640"},
    {"name": "Chambray Button-Down",       "brand": "CoastLine",      "category": "shirt", "price":  58.00, "seller": 3, "image_url": "https://images.unsplash.com/photo-1622470953794-aa9c70b0fb9d?w=640"},
    # ── Pants ─────────────────────────────────────────────────────────────────
    {"name": "Slim-Fit Blue Jeans",        "brand": "DenimWorks",     "category": "pants", "price":  89.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=640"},
    {"name": "Black Skinny Jeans",         "brand": "DenimWorks",     "category": "pants", "price":  79.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=640"},
    {"name": "Khaki Chino Pants",          "brand": "UrbanBasics",    "category": "pants", "price":  69.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=640"},
    {"name": "Grey Slim Trousers",         "brand": "OfficePro",      "category": "pants", "price":  94.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1563453392212-326f5e854473?w=640"},
    {"name": "Navy Cargo Pants",           "brand": "OutdoorCo",      "category": "pants", "price":  74.99, "seller": 3, "image_url": "https://images.unsplash.com/photo-1598554747436-c9293d6a588f?w=640"},
    {"name": "White Wide-Leg Trousers",    "brand": "SummerDays",     "category": "pants", "price":  82.00, "seller": 1, "image_url": "https://images.unsplash.com/photo-1631729371254-42c2892f0e6e?w=640"},
    {"name": "Burgundy Corduroy Pants",    "brand": "VintageAtelier", "category": "pants", "price": 110.00, "seller": 2, "image_url": "https://images.unsplash.com/photo-1519058082700-08a0b56da9b4?w=640"},
    {"name": "Olive Linen Trousers",       "brand": "CoastLine",      "category": "pants", "price":  86.00, "seller": 3, "image_url": "https://images.unsplash.com/photo-1604176354204-9268737828e4?w=640"},
    # ── Shoes ─────────────────────────────────────────────────────────────────
    {"name": "White Canvas Sneakers",      "brand": "PacificKicks",   "category": "shoes", "price":  79.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=640"},
    {"name": "Classic Running Shoes",      "brand": "SportEdge",      "category": "shoes", "price": 129.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640"},
    {"name": "Black Leather Oxfords",      "brand": "HeritageShoe",   "category": "shoes", "price": 185.00, "seller": 4, "image_url": "https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=640"},
    {"name": "Tan Suede Chelsea Boots",    "brand": "HeritageShoe",   "category": "shoes", "price": 219.00, "seller": 4, "image_url": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=640"},
    {"name": "Navy Slip-On Loafers",       "brand": "CoastLine",      "category": "shoes", "price":  94.99, "seller": 3, "image_url": "https://images.unsplash.com/photo-1584735175315-9d5df23860e6?w=640"},
    {"name": "Grey Athletic Trainers",     "brand": "SportEdge",      "category": "shoes", "price": 115.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=640"},
    {"name": "Brown Derby Shoes",          "brand": "HeritageShoe",   "category": "shoes", "price": 162.00, "seller": 4, "image_url": "https://images.unsplash.com/photo-1543512214-318c7553f230?w=640"},
    {"name": "Red High-Top Sneakers",      "brand": "PacificKicks",   "category": "shoes", "price":  99.00, "seller": 1, "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=640"},
    # ── Jackets ───────────────────────────────────────────────────────────────
    {"name": "Leather Biker Jacket",       "brand": "UrbanBasics",    "category": "jacket", "price": 199.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=640"},
    {"name": "Light Denim Jacket",         "brand": "DenimWorks",     "category": "jacket", "price":  89.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1576871337632-b9aef4c17ab9?w=640"},
    {"name": "Olive Parka Jacket",         "brand": "OutdoorCo",      "category": "jacket", "price": 149.99, "seller": 3, "image_url": "https://images.unsplash.com/photo-1544022613-e87ca75a784a?w=640"},
    {"name": "Navy Wool Blazer",           "brand": "OfficePro",      "category": "jacket", "price": 245.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=640"},
    {"name": "Caramel Trench Coat",        "brand": "VintageAtelier", "category": "jacket", "price": 320.00, "seller": 2, "image_url": "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=640"},
    # ── Dresses ───────────────────────────────────────────────────────────────
    {"name": "Floral Midi Dress",          "brand": "SummerDays",     "category": "dress",  "price": 119.00, "seller": 1, "image_url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=640"},
    {"name": "Black Bodycon Dress",        "brand": "UrbanBasics",    "category": "dress",  "price":  75.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=640"},
    {"name": "White Linen Sun Dress",      "brand": "SummerDays",     "category": "dress",  "price":  95.00, "seller": 1, "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=640"},
    {"name": "Emerald Wrap Dress",         "brand": "VintageAtelier", "category": "dress",  "price": 155.00, "seller": 2, "image_url": "https://images.unsplash.com/photo-1618932260643-eee4a2f652a6?w=640"},
    {"name": "Dusty Pink Maxi Dress",      "brand": "CoastLine",      "category": "dress",  "price": 138.00, "seller": 3, "image_url": "https://images.unsplash.com/photo-1614252369475-531eba835eb1?w=640"},
    # ── Skirts ────────────────────────────────────────────────────────────────
    {"name": "Denim Mini Skirt",           "brand": "DenimWorks",     "category": "skirt",  "price":  54.99, "seller": 1, "image_url": "https://images.unsplash.com/photo-1583496661160-fb5218ees80e?w=640"},
    {"name": "Pleated Midi Skirt",         "brand": "OfficePro",      "category": "skirt",  "price":  79.99, "seller": 0, "image_url": "https://images.unsplash.com/photo-1592301933927-35b597393c0a?w=640"},
    {"name": "Black Faux-Leather Skirt",   "brand": "UrbanBasics",    "category": "skirt",  "price": 119.00, "seller": 0, "image_url": "https://images.unsplash.com/photo-1571513722275-4b41940f54b8?w=640"},
    {"name": "Floral Wrap Skirt",          "brand": "SummerDays",     "category": "skirt",  "price":  64.00, "seller": 1, "image_url": "https://images.unsplash.com/photo-1562114808-b4b33cf4a88b?w=640"},
    {"name": "Tartan Plaid Kilt",          "brand": "VintageAtelier", "category": "skirt",  "price":  89.00, "seller": 2, "image_url": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=640"},
]
# fmt: on


# ── Ingestion helpers ──────────────────────────────────────────────────────────

async def _download_image(url: str, timeout: float = 10.0):
    """Download a URL and return a PIL Image, or None on failure."""
    import httpx
    from PIL import Image

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as exc:
        return None, str(exc)


def _embed_batch(images, encoder):
    """Synchronous CLIP encode for a batch of PIL images."""

    return encoder.encode(images)


async def _seed(args) -> None:
    """Main async seeding logic."""
    import numpy as np
    from sqlalchemy import select, text

    from app.core.database import AsyncSessionLocal, engine
    from app.models.orm import Base, Product, Seller
    from ml.clip_encoder import get_clip_encoder

    print("\n" + "=" * 62)
    print("  AI Clothing Recommender — Catalog Seeder")
    print("=" * 62)

    # ── Init DB ───────────────────────────────────────────────────────────
    print("\n[1/5] Connecting to PostgreSQL …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("      DB tables ready.")

    # ── Init Pinecone ─────────────────────────────────────────────────────
    print("\n[2/5] Connecting to Pinecone …")
    from app.services.vector_store import get_vector_service
    vector_svc = get_vector_service()
    print(f"      Index: {vector_svc._settings.pinecone_index_name}")

    # ── Optional reset ────────────────────────────────────────────────────
    if args.reset:
        print("\n      --reset: deleting all products and sellers …")
        async with AsyncSessionLocal() as session:
            await session.execute(text("DELETE FROM products"))
            await session.execute(text("DELETE FROM sellers"))
            await session.commit()
        print("      Rows cleared.")

    # ── Load CLIP ─────────────────────────────────────────────────────────
    if not args.dry_run:
        print("\n[3/5] Loading CLIP encoder …")
        encoder = get_clip_encoder()
        print("      CLIP ready.")
    else:
        encoder = None
        print("\n[3/5] Dry-run: skipping CLIP load.")

    # ── Seed sellers ──────────────────────────────────────────────────────
    print("\n[4/5] Seeding sellers …")
    seller_ids: list[uuid.UUID] = []

    async with AsyncSessionLocal() as session:
        for s in SELLERS:
            # Idempotency: skip if seller with same name already exists
            existing = await session.execute(
                select(Seller).where(Seller.name == s["name"])
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                seller_ids.append(row.id)
                print(f"      [skip]  {s['name']} (already exists)")
                continue

            obj = Seller(
                name=s["name"],
                city=s["city"],
                country=s["country"],
                latitude=s["latitude"],
                longitude=s["longitude"],
                website=s.get("website"),
            )
            session.add(obj)
            await session.flush()
            seller_ids.append(obj.id)
            print(f"      [added] {s['name']} ({s['city']})")

        await session.commit()

    print(f"      {len(seller_ids)} seller(s) ready.")

    # ── Seed products in batches ──────────────────────────────────────────
    print(f"\n[5/5] Seeding {len(CATALOG)} products (batch={args.batch}) …\n")

    succeeded = 0
    skipped   = 0
    failed    = 0

    # Process in embedding batches
    batch_size = args.batch
    for batch_start in range(0, len(CATALOG), batch_size):
        batch = CATALOG[batch_start : batch_start + batch_size]

        # ── 1. Download images ─────────────────────────────────────────
        images   : list       = []
        meta_rows: list[dict] = []

        for item in batch:
            print(f"  Downloading  {item['name']!r} … ", end="", flush=True)
            result = await _download_image(item["image_url"])
            if isinstance(result, tuple):
                img, err = result
            else:
                img, err = result, None

            if img is None:
                print(f"FAILED ({err})")
                failed += 1
                continue

            print("ok")
            images.append(img)
            meta_rows.append(item)

        if not images:
            continue

        if args.dry_run:
            print(f"  [dry-run] Would embed+upsert {len(images)} product(s).")
            skipped += len(images)
            continue

        # ── 2. Batch-embed ─────────────────────────────────────────────
        vectors: np.ndarray = await asyncio.get_event_loop().run_in_executor(
            None, _embed_batch, images, encoder
        )

        # ── 3. Write to DB + Pinecone ──────────────────────────────────
        async with AsyncSessionLocal() as session:
            for item, vector in zip(meta_rows, vectors, strict=False):
                # Idempotency: skip if product with same name already exists
                existing = await session.execute(
                    select(Product).where(Product.name == item["name"])
                )
                if existing.scalar_one_or_none() is not None:
                    print(f"  [skip]   {item['name']!r} (already in DB)")
                    skipped += 1
                    continue

                seller_id = seller_ids[item["seller"]]
                product = Product(
                    name=item["name"],
                    brand=item["brand"],
                    category=item["category"],
                    price=item["price"],
                    currency="USD",
                    image_url=item["image_url"],
                    seller_id=seller_id,
                )
                session.add(product)
                await session.flush()

                vector_id = str(product.id)
                metadata = {
                    "price":     float(item["price"]),
                    "currency":  "USD",
                    "brand":     item["brand"],
                    "category":  item["category"],
                    "name":      item["name"],
                    "seller_id": str(seller_id),
                    "image_url": item["image_url"],
                }
                vector_svc._index.upsert(
                    vectors=[{"id": vector_id, "values": vector.tolist(), "metadata": metadata}],
                    namespace=item["category"],
                )

                product.vector_id = vector_id
                print(f"  [seeded] {item['name']!r}  ${item['price']:.2f}  [{item['category']}]")
                succeeded += 1

            await session.commit()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  Done.  Seeded={succeeded}  Skipped={skipped}  Failed={failed}")
    print("=" * 62)
    print("\nNext steps:")
    print("  1. Start the API:  uvicorn app.main:app --reload")
    print("  2. Test it:        curl -X POST http://localhost:8000/api/v1/pipeline/recommend \\")
    print("                          -F 'file=@your_outfit.jpg'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the AI clothing recommender catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing products and sellers before seeding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download images and verify URLs without writing to DB or Pinecone.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        metavar="N",
        help="CLIP embedding batch size (default: 8).  "
             "Increase on GPU, decrease if OOM.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    asyncio.run(_seed(args))
    print(f"  Total time: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
