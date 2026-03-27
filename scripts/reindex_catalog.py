#!/usr/bin/env python3
"""
scripts/reindex_catalog.py
===========================
Re-embed every product in the catalog using the *currently configured* CLIP
model and re-upsert all vectors to Pinecone.

Run this whenever the CLIP model changes (e.g. Phase 8: openai/clip-vit-base-
patch32 → patrickjohncyh/fashion-clip) so that stored embeddings and live
inference use the same vector space.

The script is **non-destructive**:
  - It overwrites Pinecone vectors in-place (same ID, same namespace).
  - It does NOT modify any PostgreSQL rows.
  - If an image fails to load, the product is skipped (old vector kept).

Usage
-----
    # Re-index all products with the configured model
    python3.11 scripts/reindex_catalog.py

    # Dry-run: shows what would be re-indexed without touching Pinecone
    python3.11 scripts/reindex_catalog.py --dry-run

    # Override embedding batch size (default 8)
    python3.11 scripts/reindex_catalog.py --batch 16
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


async def _load_image_from_url(url: str, client) -> "Image.Image | None":
    """Download an image from an HTTP/HTTPS URL."""
    from PIL import Image
    try:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return None


def _load_image_from_path(path: str) -> "Image.Image | None":
    """Load an image from a local filesystem path."""
    from PIL import Image, UnidentifiedImageError
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, Exception) as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return None


async def _load_image(url_or_path: str, http_client) -> "Image.Image | None":
    """Load image from either an HTTP URL or a local path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return await _load_image_from_url(url_or_path, http_client)
    return _load_image_from_path(url_or_path)


async def _reindex(args: argparse.Namespace) -> None:
    import httpx
    import numpy as np
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.core.database import AsyncSessionLocal
    from app.models.orm import Product, Seller
    from app.services.vector_store import get_vector_service
    from ml.clip_encoder import get_clip_encoder

    settings = get_settings()

    print("\n" + "═" * 66)
    print("  AI Clothing Recommender — Catalog Re-indexer  (Phase 8)")
    print("═" * 66)
    print(f"\n  Model  : {settings.clip_model_name}")
    print(f"  Device : {settings.clip_device}")
    print(f"  Index  : {settings.pinecone_index_name}")

    # ── Load products from PostgreSQL ──────────────────────────────────────
    print("\n[1/4] Fetching products from PostgreSQL …")
    from sqlalchemy.orm import selectinload
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Product)
            .where(Product.image_url.isnot(None))
            .options(selectinload(Product.seller))
            .order_by(Product.category)
        )
        products = result.scalars().all()

    if not products:
        print("      ✗  No products with image_url found in the database.")
        return
    print(f"      {len(products)} products found.")

    # ── Load CLIP encoder (picks up new model from config/env) ─────────────
    print(f"\n[2/4] Loading CLIP encoder '{settings.clip_model_name}' …")
    encoder = get_clip_encoder()
    print("      CLIP ready.")

    # ── Connect to Pinecone ────────────────────────────────────────────────
    print("\n[3/4] Connecting to Pinecone …")
    vector_svc = get_vector_service()
    print(f"      Index: {settings.pinecone_index_name}")

    if args.dry_run:
        print("\n  Dry-run mode — no writes will be made to Pinecone.\n")

    # ── Re-embed + upsert in batches ───────────────────────────────────────
    print(f"\n[4/4] Re-indexing {len(products)} products (batch={args.batch}) …\n")

    succeeded = skipped = failed = 0
    loop = asyncio.get_event_loop()

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
        for batch_start in range(0, len(products), args.batch):
            batch = products[batch_start : batch_start + args.batch]

            # ── Load images concurrently ───────────────────────────────────
            load_tasks = [
                _load_image(p.image_url, http_client)
                for p in batch
            ]
            images = await asyncio.gather(*load_tasks)

            # Pair products with their loaded images, filter failures
            valid_pairs = [
                (prod, img)
                for prod, img in zip(batch, images, strict=False)
                if img is not None
            ]
            batch_failed = len(batch) - len(valid_pairs)
            failed += batch_failed

            if batch_failed:
                for prod, img in zip(batch, images, strict=False):
                    if img is None:
                        log.warning("  [skip]  %r — image failed to load", prod.name)

            if not valid_pairs:
                continue

            if args.dry_run:
                print(
                    f"  [dry-run]  {batch_start:>4}–{batch_start + len(batch) - 1:<4} "
                    f"ok={len(valid_pairs)} failed={batch_failed}"
                )
                skipped += len(valid_pairs)
                continue

            valid_prods, valid_imgs = zip(*valid_pairs, strict=False)

            # ── Batch CLIP encode off the event loop ───────────────────────
            vectors: np.ndarray = await loop.run_in_executor(
                None, encoder.encode, list(valid_imgs)
            )

            # ── Drop any vectors that are still non-finite after encoding ─────────
            # FashionCLIP can return NaN/Inf for heavily corrupted or
            # near-blank images even after the norm-clamp in the encoder;
            # Pinecone rejects those with a 400 "Unexpected token".
            finite_pairs: list[tuple] = []
            for prod, vec in zip(valid_prods, vectors, strict=False):
                if not np.isfinite(vec).all():
                    log.warning(
                        "  [skip]  %r — non-finite embedding (NaN/Inf), discarding",
                        prod.name,
                    )
                    failed += 1
                else:
                    finite_pairs.append((prod, vec))

            # ── Build Pinecone upsert payload ──────────────────────────────────────
            upsert_payload: list[dict] = []
            for prod, vector in finite_pairs:
                if not prod.vector_id:
                    log.warning("  [skip]  %r — no vector_id, not yet indexed", prod.name)
                    skipped += 1
                    continue

                # Rebuild full metadata from DB — ensures Phase 7 fields are
                # included even for products seeded before the schema change.
                metadata: dict = {
                    "price":     float(prod.price),
                    "currency":  prod.currency or "USD",
                    "brand":     prod.brand or "",
                    "category":  prod.category,
                    "name":      prod.name,
                    "seller_id": str(prod.seller_id) if prod.seller_id else "",
                    "image_url": prod.image_url or "",
                    # Phase 7 extended attributes (empty string = no filter match)
                    "colour":    prod.colour   or "",
                    "gender":    prod.gender   or "",
                    "style":     prod.style    or "",
                    "occasion":  prod.occasion or "",
                }

                upsert_payload.append({
                    "id":        prod.vector_id,
                    "values":    vector.tolist(),
                    "namespace": prod.category,
                    "metadata":  metadata,
                })

            if upsert_payload:
                vector_svc.upsert(upsert_payload)
                succeeded += len(upsert_payload)

            # Progress heartbeat
            batch_end = batch_start + len(batch)
            pct = 100 * batch_end / len(products)
            names = ", ".join(f"'{p.name}'" for p in valid_prods[:2])
            suffix = "…" if len(valid_prods) > 2 else ""
            print(f"  ✓ [{batch_end:>3}/{len(products)}] {pct:>3.0f}%   {names}{suffix}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print(f"  Re-indexed : {succeeded}")
    print(f"  Skipped    : {skipped}  (no vector_id or dry-run)")
    print(f"  Failed     : {failed}   (image load errors)")
    print("═" * 66)

    if args.dry_run or succeeded == 0:
        return

    # ── Verify Pinecone counts (unchanged — only values updated in-place) ──
    print("\n  Pinecone namespace summary (counts should be unchanged):")
    try:
        stats = vector_svc._index.describe_index_stats()
        ns_data = stats.get("namespaces", {})
        total = stats.get("total_vector_count", 0)
        for ns in sorted(ns_data):
            count = ns_data[ns].get("vector_count", 0)
            print(f"    {ns:<10}  {count:>5} vectors")
        print(f"    {'TOTAL':<10}  {total:>5} vectors")
    except Exception as exc:
        log.warning("Could not fetch Pinecone stats: %s", exc)

    print(
        f"\n  ✅  All vectors now use '{settings.clip_model_name}'.\n"
        "     Restart uvicorn so the live server also loads the new model:\n\n"
        "       kill $(lsof -ti :8000)\n"
        "       python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed all catalog products with the configured CLIP model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        metavar="N",
        help="CLIP embedding batch size (default: 8). Increase on GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-indexed without writing to Pinecone.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    asyncio.run(_reindex(args))
    print(f"  Total elapsed: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
