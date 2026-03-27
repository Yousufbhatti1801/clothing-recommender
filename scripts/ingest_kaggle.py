#!/usr/bin/env python3
"""
scripts/ingest_kaggle.py
=========================
ETL pipeline: Kaggle "Fashion Product Images Small" → PostgreSQL + Pinecone.

Reads the dataset CSV, samples up to ``--per-category`` products per garment
class, loads images from disk, batch-encodes them with CLIP, inserts enriched
rows into PostgreSQL, and upserts the vectors (with extended metadata) into
Pinecone — reaching the Phase 7 target of 1,000+ catalog items.

Prerequisites
─────────────
1. Download and extract the Kaggle dataset:

       pip install kaggle
       kaggle datasets download -d paramaggarwal/fashion-product-images-small
       mkdir -p data/kaggle_fashion_small
       unzip fashion-product-images-small.zip -d data/kaggle_fashion_small/

   Or download manually from:
   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

2. Make sure the DB schema is up to date (Phase 7 columns):
   The script runs the migration inline, OR you can run it separately:

       python3.11 scripts/migrate_add_product_attrs.py

Usage
─────
    # Basic (1 000 products balanced across 6 categories)
    python3.11 scripts/ingest_kaggle.py --data-dir data/kaggle_fashion_small

    # More products, larger CLIP batch on GPU
    python3.11 scripts/ingest_kaggle.py --data-dir data/kaggle_fashion_small \\
        --limit 2000 --per-category 334 --batch 32

    # Dry-run: validate CSV + images without writing anything
    python3.11 scripts/ingest_kaggle.py --data-dir data/kaggle_fashion_small --dry-run

    # Re-ingest after a reset (removes all kaggle:// sourced products first)
    python3.11 scripts/ingest_kaggle.py --data-dir data/kaggle_fashion_small --reset
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Category mapping ───────────────────────────────────────────────────────────
# Maps Kaggle ``articleType`` → our six GarmentCategory string values.
# Source: exploratory analysis of the styles.csv articleType distribution.
# Only article types listed here are ingested; everything else is skipped.
_CATEGORY_MAP: dict[str, str] = {
    # Tops / Shirts
    "Tshirts":       "shirt",
    "Shirts":        "shirt",
    "Tops":          "shirt",
    "Sweatshirts":   "shirt",
    "Sweaters":      "shirt",
    "Kurtas":        "shirt",
    "Kurta Sets":    "shirt",
    # Outerwear / Jackets
    "Jackets":       "jacket",
    "Blazers":       "jacket",
    "Waistcoat":     "jacket",
    "Windcheater":   "jacket",
    # Bottoms / Pants
    "Jeans":                    "pants",
    "Casual Trousers":          "pants",
    "Formal Trousers":          "pants",
    "Track Pants & Joggers":    "pants",
    "Shorts":                   "pants",
    "Leggings":                 "pants",
    "Capris":                   "pants",
    "Patiala":                  "pants",
    "Salwar":                   "pants",
    # Footwear / Shoes
    "Casual Shoes":     "shoes",
    "Formal Shoes":     "shoes",
    "Sports Shoes":     "shoes",
    "Boots":            "shoes",
    "Sandals":          "shoes",
    "Flats":            "shoes",
    "Heels":            "shoes",
    "Flip Flops":       "shoes",
    "Sports Sandals":   "shoes",
    # Dresses
    "Dresses":      "dress",
    "Jumpsuit":     "dress",
    "Rompers":      "dress",
    "Night suits":  "dress",
    # Skirts
    "Skirts": "skirt",
}

# Gender normalisation
_GENDER_MAP: dict[str, str] = {
    "Men":    "men",
    "Women":  "women",
    "Unisex": "unisex",
    "Boys":   "men",
    "Girls":  "women",
}

# Kaggle ``usage`` → our style label
_STYLE_MAP: dict[str, str] = {
    "Casual":       "casual",
    "Formal":       "formal",
    "Sports":       "sportswear",
    "Ethnic":       "ethnic",
    "Party":        "party",
    "Smart Casual": "smart casual",
    "Travel":       "casual",
    "Home":         "casual",
    "NA":           "casual",
}

# Deterministic price ranges per category (USD)
_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "shirt":  (15.0,  80.0),
    "pants":  (30.0, 120.0),
    "shoes":  (40.0, 250.0),
    "jacket": (50.0, 350.0),
    "dress":  (40.0, 200.0),
    "skirt":  (25.0, 120.0),
}

# These are the sellers already seeded by seed_catalog.py.
# Kaggle products are distributed across them round-robin.
_SELLER_NAMES: list[str] = [
    "Urban Threads NYC",
    "Pacific Style LA",
    "Vintage Atelier London",
    "CoastLine Seattle",
    "Heritage Shoe Boston",
]


# ── Internal data class ────────────────────────────────────────────────────────

class _KaggleProduct(NamedTuple):
    """One parsed and normalised Kaggle product ready for ingestion."""

    kaggle_id: str           # original numeric ID from styles.csv
    name: str                # productDisplayName (truncated to 255)
    brand: str               # heuristic: first word of display name
    category: str            # one of our 6 GarmentCategory values
    colour: str | None       # baseColour, lowercased
    gender: str              # "men" | "women" | "unisex"
    style: str               # normalised from usage column
    occasion: str            # raw usage, lowercased
    price: float             # deterministically generated from ID + category
    image_path: Path         # absolute path to the local JPEG


# ── Helpers ────────────────────────────────────────────────────────────────────

def _deterministic_price(kaggle_id: str, category: str) -> float:
    """Generate a stable price based on product ID + category."""
    low, high = _PRICE_RANGES.get(category, (20.0, 100.0))
    seed = int(kaggle_id) if kaggle_id.isdigit() else hash(kaggle_id)
    return round(random.Random(seed).uniform(low, high), 2)


def _extract_brand(display_name: str) -> str:
    """
    Best-effort brand extraction from ``productDisplayName``.

    The first word of the Kaggle display name is almost always the brand
    (e.g. "Roadster Men's Blue Casual Shirt" → "Roadster").
    Generic non-brand words are filtered out.
    """
    _skip = {"men", "women", "boys", "girls", "unisex", "the", "a", "an"}
    words = display_name.strip().split()
    if not words:
        return "Unknown"
    first = words[0].rstrip("'\"")
    if first.lower() in _skip and len(words) > 1:
        return words[1].rstrip("'\"")
    return first


def _load_and_sample(
    data_dir: Path,
    per_category: int,
    total_limit: int,
) -> list[_KaggleProduct]:
    """
    Read ``styles.csv``, map to our schema, and sample up to ``per_category``
    items per GarmentCategory (capped at ``total_limit`` total).

    Returns a flat list sorted by category for reproducible ordering.
    Rows with missing images or unmapped article types are silently skipped.
    """
    styles_csv = data_dir / "styles.csv"
    images_dir = data_dir / "images"

    if not styles_csv.exists():
        raise FileNotFoundError(
            f"styles.csv not found at {styles_csv}\n\n"
            "Download the dataset:\n"
            "  kaggle datasets download -d paramaggarwal/fashion-product-images-small\n"
            "  unzip fashion-product-images-small.zip -d data/kaggle_fashion_small/"
        )

    log.info("Reading %s …", styles_csv)

    # Separate quota bucket per category
    buckets: dict[str, list[_KaggleProduct]] = {c: [] for c in _PRICE_RANGES}

    with styles_csv.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            article_type = (row.get("articleType") or "").strip()
            category = _CATEGORY_MAP.get(article_type)
            if category is None:
                continue

            bucket = buckets.get(category, [])
            if len(bucket) >= per_category:
                continue

            kaggle_id = (row.get("id") or "").strip()
            if not kaggle_id:
                continue

            image_path = images_dir / f"{kaggle_id}.jpg"
            if not image_path.exists():
                continue  # image missing from dataset → skip

            display_name = (row.get("productDisplayName") or "").strip()
            if not display_name:
                display_name = f"{article_type} {kaggle_id}"

            colour_raw = (row.get("baseColour") or "").strip()
            usage_raw = (row.get("usage") or "Casual").strip()

            bucket.append(
                _KaggleProduct(
                    kaggle_id=kaggle_id,
                    name=display_name[:255],
                    brand=_extract_brand(display_name),
                    category=category,
                    colour=colour_raw.lower() if colour_raw else None,
                    gender=_GENDER_MAP.get((row.get("gender") or "").strip(), "unisex"),
                    style=_STYLE_MAP.get(usage_raw, "casual"),
                    occasion=usage_raw.lower() if usage_raw != "NA" else "casual",
                    price=_deterministic_price(kaggle_id, category),
                    image_path=image_path,
                )
            )

    # Merge all buckets and apply total limit
    all_products: list[_KaggleProduct] = []
    for cat in sorted(buckets):
        all_products.extend(buckets[cat])
    all_products = all_products[:total_limit]

    per_cat_counts = {
        cat: sum(1 for p in all_products if p.category == cat)
        for cat in _PRICE_RANGES
    }
    log.info("Sampled %d products: %s", len(all_products), per_cat_counts)
    return all_products


def _load_image_safe(path: Path) -> Image.Image | None:
    """Load one image from disk; return None on any I/O or format error."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, Exception):
        return None


# ── Main ingestion coroutine ───────────────────────────────────────────────────

async def _ingest(args: argparse.Namespace) -> None:
    """Full ETL: CSV → images → CLIP → PostgreSQL + Pinecone."""
    from sqlalchemy import select, text

    from app.core.database import AsyncSessionLocal, engine
    from app.models.orm import Base, Product, Seller
    from app.services.vector_store import get_vector_service
    from ml.clip_encoder import get_clip_encoder

    print("\n" + "═" * 66)
    print("  AI Clothing Recommender — Kaggle Catalog Ingestor  (Phase 7)")
    print("═" * 66)

    data_dir = Path(args.data_dir).expanduser().resolve()

    # ── [1/6] Load and sample CSV ──────────────────────────────────────────
    print(f"\n[1/6] Reading Kaggle dataset from: {data_dir}")
    products_to_ingest = _load_and_sample(
        data_dir=data_dir,
        per_category=args.per_category,
        total_limit=args.limit,
    )
    if not products_to_ingest:
        print("      ✗  No mappable products found. Check --data-dir and styles.csv.")
        return
    print(f"      {len(products_to_ingest)} products sampled.")

    # ── [2/6] Ensure DB schema (idempotent migration) ──────────────────────
    print("\n[2/6] Connecting to PostgreSQL and ensuring schema …")
    async with engine.begin() as conn:
        # Create tables for fresh installs
        await conn.run_sync(Base.metadata.create_all)
        # Add new attribute columns to existing tables (no-op if already present)
        for stmt in [
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS colour VARCHAR(50)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS material VARCHAR(100)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS style VARCHAR(100)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS gender VARCHAR(20)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS size_available TEXT",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS occasion VARCHAR(100)",
            "CREATE INDEX IF NOT EXISTS ix_products_category ON products (category)",
            "CREATE INDEX IF NOT EXISTS ix_products_category_price ON products (category, price)",
            "CREATE INDEX IF NOT EXISTS ix_products_colour ON products (colour)",
        ]:
            await conn.execute(text(stmt))
    print("      DB schema ready.")

    # ── [3/6] Connect to Pinecone ──────────────────────────────────────────
    print("\n[3/6] Connecting to Pinecone …")
    vector_svc = get_vector_service()
    index_name = vector_svc._settings.pinecone_index_name
    print(f"      Index: {index_name}")

    # ── [4/6] Optional reset (Kaggle-sourced products only) ───────────────
    if args.reset:
        print("\n      --reset: removing previously Kaggle-ingested products …")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Product).where(
                    Product.product_url.like("kaggle://fashion-small/%")
                )
            )
            stale = result.scalars().all()
            removed = 0
            for p in stale:
                if p.vector_id:
                    try:
                        vector_svc._index.delete(ids=[p.vector_id], namespace=p.category)
                    except Exception as exc:
                        log.warning("Could not delete vector %s: %s", p.vector_id, exc)
                await session.delete(p)
                removed += 1
            await session.commit()
        print(f"      Removed {removed} stale product(s).")

    # ── [5/6] Load CLIP encoder ────────────────────────────────────────────
    if not args.dry_run:
        print("\n[4/6] Loading CLIP encoder (this may take a moment on first run) …")
        encoder = get_clip_encoder()
        print("      CLIP ready.")
    else:
        encoder = None
        print("\n[4/6] Dry-run mode — skipping CLIP load.")

    # ── [5/6] Resolve seller pool ──────────────────────────────────────────
    print("\n[5/6] Resolving seller pool …")
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Seller).where(Seller.name.in_(_SELLER_NAMES))
        )
        sellers = result.scalars().all()

    if not sellers:
        print(
            "      ✗  No sellers found in the database.\n"
            "         Run 'python3.11 db/seed.py' or 'python3.11 scripts/seed_catalog.py' first."
        )
        return
    print(f"      {len(sellers)} seller(s) available for assignment.")

    # ── [6/6] Batch ingest ─────────────────────────────────────────────────
    print(
        f"\n[6/6] Ingesting {len(products_to_ingest)} products  "
        f"[batch={args.batch}, workers={args.workers}] …\n"
    )

    succeeded = skipped = failed = 0
    loop = asyncio.get_event_loop()
    image_executor = ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="img_load",
    )

    try:
        for batch_start in range(0, len(products_to_ingest), args.batch):
            batch = products_to_ingest[batch_start : batch_start + args.batch]

            # ── Parallel image loading (I/O bound → thread pool) ───────────
            load_tasks = [
                loop.run_in_executor(image_executor, _load_image_safe, p.image_path)
                for p in batch
            ]
            loaded_images: list[Image.Image | None] = await asyncio.gather(*load_tasks)

            # Filter out products whose images failed to load
            valid_pairs = [
                (prod, img)
                for prod, img in zip(batch, loaded_images, strict=False)
                if img is not None
            ]
            batch_failed = len(batch) - len(valid_pairs)
            failed += batch_failed

            if not valid_pairs:
                continue

            if args.dry_run:
                print(
                    f"  [dry-run]  batch {batch_start:>4}–{batch_start + len(batch) - 1:<4}  "
                    f"ok={len(valid_pairs)}  failed={batch_failed}"
                )
                skipped += len(valid_pairs)
                continue

            valid_prods, valid_imgs = zip(*valid_pairs, strict=False)

            # ── Batch CLIP encoding (CPU/GPU bound → default executor) ─────
            # encoder.encode() is synchronous; run off the event loop so we
            # don't block other async tasks.
            vectors: np.ndarray = await loop.run_in_executor(
                None,
                encoder.encode,
                list(valid_imgs),
            )

            # ── DB insert + build Pinecone batch ───────────────────────────
            pinecone_upsert: list[dict] = []

            async with AsyncSessionLocal() as session:
                for prod, vector in zip(valid_prods, vectors, strict=False):
                    # Deduplication: use product_url as a stable source key.
                    # kaggle://fashion-small/{id} is unique per Kaggle product.
                    source_url = f"kaggle://fashion-small/{prod.kaggle_id}"

                    existing = await session.execute(
                        select(Product).where(Product.product_url == source_url)
                    )
                    if existing.scalar_one_or_none() is not None:
                        skipped += 1
                        continue

                    # Deterministic seller assignment (round-robin by kaggle_id hash)
                    seller = sellers[hash(prod.kaggle_id) % len(sellers)]

                    # Insert product with all Phase 7 attributes
                    db_product = Product(
                        name=prod.name,
                        brand=prod.brand,
                        category=prod.category,
                        price=prod.price,
                        currency="USD",
                        image_url=str(prod.image_path),
                        product_url=source_url,      # ← dedup key
                        colour=prod.colour,
                        style=prod.style,
                        gender=prod.gender,
                        occasion=prod.occasion,
                        seller_id=seller.id,
                    )
                    session.add(db_product)
                    await session.flush()  # assigns db_product.id

                    vector_id = str(db_product.id)
                    db_product.vector_id = vector_id

                    # Build flat Pinecone metadata — all values are scalars
                    # (no nested dicts) so every field is filterable server-side.
                    pinecone_upsert.append({
                        "id":        vector_id,
                        "values":    vector.tolist(),
                        "namespace": prod.category,
                        "metadata": {
                            "price":     float(prod.price),
                            "currency":  "USD",
                            "brand":     prod.brand,
                            "category":  prod.category,
                            "name":      prod.name,
                            "seller_id": str(seller.id),
                            "image_url": str(prod.image_path),
                            # Phase 7 extended fields — enable Pinecone server-side filters
                            "colour":   prod.colour or "",
                            "gender":   prod.gender,
                            "style":    prod.style,
                            "occasion": prod.occasion,
                        },
                    })

                    succeeded += 1

                await session.commit()

            # ── Upsert this batch to Pinecone (grouped by namespace inside) ─
            if pinecone_upsert:
                vector_svc.upsert(pinecone_upsert)

            # Progress heartbeat every 100 products
            batch_end = batch_start + len(batch)
            if succeeded % 100 < args.batch or batch_end >= len(products_to_ingest):
                pct = 100 * (batch_end) / len(products_to_ingest)
                print(
                    f"  ✓ {succeeded:>4} ingested  "
                    f"skipped={skipped}  failed={failed}  "
                    f"({pct:.0f}%)"
                )

    finally:
        image_executor.shutdown(wait=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print(f"  Ingested : {succeeded}")
    print(f"  Skipped  : {skipped}  (already in DB)")
    print(f"  Failed   : {failed}   (image load errors)")
    print("═" * 66)

    if args.dry_run or succeeded == 0:
        return

    # ── Verify Pinecone counts ─────────────────────────────────────────────
    print("\n  Pinecone namespace summary:")
    try:
        stats = vector_svc._index.describe_index_stats()
        ns_data = stats.get("namespaces", {})
        for ns in sorted(_PRICE_RANGES):
            count = ns_data.get(ns, {}).get("vector_count", 0)
            bar = "█" * (count // 10)
            print(f"    {ns:<8}  {count:>5} vectors  {bar}")
        total = stats.get("total_vector_count", 0)
        print(f"    {'TOTAL':<8}  {total:>5} vectors")
    except Exception as exc:
        log.warning("Could not fetch Pinecone stats: %s", exc)

    print("\n  Next steps:")
    print("    1. Start the API:  python3.11 -m uvicorn app.main:app --reload")
    print("    2. Test it:        curl -X POST http://localhost:8000/api/v1/pipeline/recommend \\")
    print("                            -F 'file=@your_outfit.jpg'")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Kaggle Fashion Product Images Small into the catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="data/kaggle_fashion_small",
        metavar="PATH",
        help="Path to the extracted Kaggle dataset (default: data/kaggle_fashion_small)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum total products to ingest (default: 1000)",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=167,
        metavar="N",
        help="Maximum products per category (default: 167 → ~1000 across 6 cats)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        metavar="N",
        help="CLIP embedding batch size (default: 16). Reduce if CPU OOM.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Thread pool workers for parallel image loading (default: 4)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all previously Kaggle-ingested products before re-running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV + images without writing to DB or Pinecone.",
    )

    args = parser.parse_args()

    t0 = time.perf_counter()
    asyncio.run(_ingest(args))
    print(f"  Total elapsed: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
