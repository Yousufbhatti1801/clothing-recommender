#!/usr/bin/env python3
"""
scripts/download_test_images.py
================================
Download real fashion images from Unsplash (free, no API key needed)
for end-to-end pipeline testing.

Images are split into two directories:
    tests/real_images/outfits/   — Full-body photos (person wearing clothes)
    tests/real_images/products/  — Individual product shots (one garment)

Usage
-----
    python3.11 scripts/download_test_images.py           # Download all
    python3.11 scripts/download_test_images.py --only outfits
    python3.11 scripts/download_test_images.py --only products
    python3.11 scripts/download_test_images.py --dry-run  # Show URLs only

Notes
-----
- All images are sourced from Unsplash (https://unsplash.com/license).
  Unsplash grants a free, irrevocable license for any purpose including
  commercial use. No attribution is legally required (but appreciated).
- Images are downloaded at 640px width for consistency with YOLO input.
- If a file already exists on disk it is skipped (idempotent).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parent.parent

# ─── Curated image URLs ──────────────────────────────────────────────────────
# Each entry: (filename, url, description, expected_garments)
# "expected_garments" is a list of categories we expect YOLO to detect.

OUTFIT_IMAGES: list[dict] = [
    {
        "filename": "outfit_casual_male.jpg",
        "url": "https://images.unsplash.com/photo-1516257984-b1b4d707412e?w=640",
        "description": "Man in casual outfit — shirt, jeans, sneakers",
        "expected": ["shirt", "pants", "shoes"],
    },
    {
        "filename": "outfit_formal_male.jpg",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640",
        "description": "Man in formal wear — blazer, shirt, trousers",
        "expected": ["shirt", "pants", "jacket"],
    },
    {
        "filename": "outfit_streetwear.jpg",
        "url": "https://images.unsplash.com/photo-1523398002811-999ca8dec234?w=640",
        "description": "Streetwear outfit — hoodie, pants, sneakers",
        "expected": ["shirt", "pants", "shoes"],
    },
    {
        "filename": "outfit_summer_dress.jpg",
        "url": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=640",
        "description": "Woman in summer dress",
        "expected": ["dress"],
    },
    {
        "filename": "outfit_business_woman.jpg",
        "url": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=640",
        "description": "Woman in business attire — blazer, skirt, heels",
        "expected": ["jacket", "skirt", "shoes"],
    },
    {
        "filename": "outfit_casual_female.jpg",
        "url": "https://images.unsplash.com/photo-1485968579580-b6d095142e6e?w=640",
        "description": "Woman in casual outfit — top, jeans, sneakers",
        "expected": ["shirt", "pants", "shoes"],
    },
    {
        "filename": "outfit_winter_jacket.jpg",
        "url": "https://images.unsplash.com/photo-1544022613-e87ca75a784a?w=640",
        "description": "Person in winter jacket and pants",
        "expected": ["jacket", "pants"],
    },
    {
        "filename": "outfit_full_body_male_2.jpg",
        "url": "https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?w=640",
        "description": "Man full body — shirt, chinos, boots",
        "expected": ["shirt", "pants", "shoes"],
    },
    {
        "filename": "outfit_athleisure.jpg",
        "url": "https://images.unsplash.com/photo-1556906781-9a412961c28c?w=640",
        "description": "Athletic outfit with sneakers",
        "expected": ["shirt", "pants", "shoes"],
    },
    {
        "filename": "outfit_skirt_outfit.jpg",
        "url": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=640",
        "description": "Woman in skirt and top",
        "expected": ["shirt", "skirt"],
    },
]

PRODUCT_IMAGES: list[dict] = [
    # Shirts / tops
    {
        "filename": "product_white_tshirt.jpg",
        "url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=640",
        "description": "White t-shirt flat lay",
        "expected": ["shirt"],
    },
    {
        "filename": "product_polo_shirt.jpg",
        "url": "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=640",
        "description": "Black polo shirt",
        "expected": ["shirt"],
    },
    {
        "filename": "product_flannel_shirt.jpg",
        "url": "https://images.unsplash.com/photo-1589310243389-96a5483213a8?w=640",
        "description": "Plaid flannel shirt",
        "expected": ["shirt"],
    },
    # Pants
    {
        "filename": "product_blue_jeans.jpg",
        "url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=640",
        "description": "Blue jeans folded",
        "expected": ["pants"],
    },
    {
        "filename": "product_chinos.jpg",
        "url": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=640",
        "description": "Khaki chinos",
        "expected": ["pants"],
    },
    # Shoes
    {
        "filename": "product_white_sneakers.jpg",
        "url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=640",
        "description": "White canvas sneakers",
        "expected": ["shoes"],
    },
    {
        "filename": "product_running_shoes.jpg",
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640",
        "description": "Red running shoe",
        "expected": ["shoes"],
    },
    {
        "filename": "product_leather_oxfords.jpg",
        "url": "https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=640",
        "description": "Black leather oxford shoes",
        "expected": ["shoes"],
    },
    # Jackets
    {
        "filename": "product_leather_jacket.jpg",
        "url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=640",
        "description": "Leather biker jacket",
        "expected": ["jacket"],
    },
    {
        "filename": "product_denim_jacket.jpg",
        "url": "https://images.unsplash.com/photo-1576871337632-b9aef4c17ab9?w=640",
        "description": "Light denim jacket",
        "expected": ["jacket"],
    },
    # Dresses
    {
        "filename": "product_floral_dress.jpg",
        "url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=640",
        "description": "Floral midi dress",
        "expected": ["dress"],
    },
    {
        "filename": "product_black_dress.jpg",
        "url": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=640",
        "description": "Black bodycon dress",
        "expected": ["dress"],
    },
    # Skirts
    {
        "filename": "product_pleated_skirt.jpg",
        "url": "https://images.unsplash.com/photo-1592301933927-35b597393c0a?w=640",
        "description": "Pleated midi skirt",
        "expected": ["skirt"],
    },
]


# ─── Download helpers ─────────────────────────────────────────────────────────

async def download_image(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Download a single image. Returns (success, message)."""
    if dest.exists():
        return True, f"  ✓ {dest.name} (already exists, skipped)"
    if dry_run:
        return True, f"  ⏩ {dest.name} → {url}"
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        # Validate it's actually an image
        img = Image.open(BytesIO(resp.content))
        img.verify()
        # Save
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        return True, f"  ✓ {dest.name} ({img.size[0]}×{img.size[1]}, {size_kb:.0f} KB)"
    except httpx.HTTPStatusError as e:
        return False, f"  ✗ {dest.name} — HTTP {e.response.status_code}"
    except Exception as e:
        return False, f"  ✗ {dest.name} — {e}"


async def download_batch(
    images: list[dict],
    dest_dir: Path,
    *,
    dry_run: bool = False,
    concurrency: int = 4,
) -> tuple[int, int]:
    """Download a batch of images with limited concurrency."""
    sem = asyncio.Semaphore(concurrency)
    ok = fail = 0

    async def _task(entry: dict):
        nonlocal ok, fail
        async with sem:
            async with httpx.AsyncClient(timeout=30.0) as client:
                dest = dest_dir / entry["filename"]
                success, msg = await download_image(client, entry["url"], dest, dry_run=dry_run)
                print(msg)
                if success:
                    ok += 1
                else:
                    fail += 1

    await asyncio.gather(*[_task(e) for e in images])
    return ok, fail


# ─── Manifest file ───────────────────────────────────────────────────────────

def write_manifest(dest_dir: Path, images: list[dict], label: str) -> None:
    """Write a JSON manifest listing downloaded images + expected detections."""
    import json
    manifest = {
        "label": label,
        "count": len(images),
        "images": [
            {
                "filename": img["filename"],
                "description": img["description"],
                "expected_garments": img["expected"],
            }
            for img in images
        ],
    }
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  📄 Wrote {manifest_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Download real fashion images for end-to-end testing",
    )
    parser.add_argument(
        "--only",
        choices=["outfits", "products"],
        help="Download only one category",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "tests" / "real_images",
        help="Base directory for downloaded images",
    )
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  Fashion Test Image Downloader")
    print("=" * 62)

    total_ok = total_fail = 0
    t0 = time.perf_counter()

    if args.only != "products":
        print(f"\n📸 Outfits ({len(OUTFIT_IMAGES)} images) → {args.dest / 'outfits'}")
        ok, fail = await download_batch(
            OUTFIT_IMAGES, args.dest / "outfits", dry_run=args.dry_run,
        )
        write_manifest(args.dest / "outfits", OUTFIT_IMAGES, "outfit_images")
        total_ok += ok
        total_fail += fail

    if args.only != "outfits":
        print(f"\n🏷️  Products ({len(PRODUCT_IMAGES)} images) → {args.dest / 'products'}")
        ok, fail = await download_batch(
            PRODUCT_IMAGES, args.dest / "products", dry_run=args.dry_run,
        )
        write_manifest(args.dest / "products", PRODUCT_IMAGES, "product_images")
        total_ok += ok
        total_fail += fail

    elapsed = time.perf_counter() - t0
    print(f"\n{'─' * 62}")
    print(f"  Done in {elapsed:.1f}s — {total_ok} downloaded, {total_fail} failed")
    if total_fail:
        print("  ⚠️  Some images failed. Re-run to retry (existing files are skipped).")
    print()


if __name__ == "__main__":
    asyncio.run(main())
