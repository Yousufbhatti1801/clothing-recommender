#!/usr/bin/env python3
"""
scripts/prepare_fashionpedia.py
================================
Download the Fashionpedia detection dataset from HuggingFace and convert
it to the app's 13-class YOLO format.

Dataset : detection-datasets/fashionpedia (45 k train + 1 158 val images)
Source  : https://huggingface.co/datasets/detection-datasets/fashionpedia

The script:
  1. Downloads fashionpedia parquet shards via HTTPS (no API key required).
  2. Reads images (embedded as bytes) and bounding boxes.
  3. Maps fashionpedia 46-class schema → app 13-class schema.
  4. Writes YOLO-format label files and JPEG images to data/fashion_dataset/.
  5. Prints a dataset summary with class distribution.

Usage
─────
    # Default: 1 train shard (~5 700 images) + full val (~1 158 images)
    python3.11 scripts/prepare_fashionpedia.py

    # More training data (2 shards ≈ 11 400 images):
    python3.11 scripts/prepare_fashionpedia.py --train-shards 2

    # Custom output directory:
    python3.11 scripts/prepare_fashionpedia.py --out-root data/fashion_dataset

    # Keep parquet cache (avoids re-download):
    python3.11 scripts/prepare_fashionpedia.py --keep-cache
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.fashion_classes import APP_CLASS_NAMES

# ── Fashionpedia HuggingFace parquet URLs ─────────────────────────────────────
_HF_BASE = (
    "https://huggingface.co/datasets/detection-datasets/fashionpedia"
    "/resolve/refs%2Fconvert%2Fparquet/default"
)
TRAIN_SHARD_URLS: list[str] = [
    f"{_HF_BASE}/train/{i:04d}.parquet" for i in range(8)
]
VAL_SHARD_URL: str = f"{_HF_BASE}/val/0000.parquet"

# ── Fashionpedia category index → app class index ────────────────────────────
# Fashionpedia has 46 classes: 0-12 whole garments, 13-25 accessories, 26+ attributes.
# We keep only the classes that map cleanly to our 13-class app schema.
FASHIONPEDIA_TO_APP: dict[int, int] = {
    0:  0,   # shirt, blouse       → shirt      (app 0)
    1:  1,   # top, t-shirt        → t-shirt     (app 1)
    2:  0,   # sweater             → shirt       (closest)
    3:  7,   # cardigan            → jacket      (app 7)
    4:  7,   # jacket              → jacket      (app 7)
    5:  7,   # vest                → jacket      (app 7)
    6:  2,   # pants               → pants       (app 2)
    7:  4,   # shorts              → shorts      (app 4)
    8:  10,  # skirt               → skirt       (app 10)
    9:  8,   # coat                → coat        (app 8)
    10: 9,   # dress               → dress       (app 9)
    11: 9,   # jumpsuit            → dress       (closest)
    12: 8,   # cape                → coat        (app 8)
    14: 12,  # hat                 → hat         (app 12)
    23: 5,   # shoe                → shoes       (app 5)
    24: 11,  # bag, wallet         → bag         (app 11)
    # 13 (glasses), 15-22, 25+ accessories/attributes → discard (-1)
}

# ── Fashionpedia category names (for diagnostics) ─────────────────────────────
FASHIONPEDIA_NAMES: list[str] = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
    "cape", "glasses", "hat", "headband", "tie", "glove", "watch", "belt",
    "leg warmer", "tights", "sock", "shoe", "bag, wallet", "scarf",
    "umbrella", "hood", "collar", "lapel", "epaulette", "sleeve", "pocket",
    "neckline", "buckle", "zipper", "applique", "bead", "bow", "flower",
    "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel",
]


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            bar.update(len(chunk))


# ─────────────────────────────────────────────────────────────────────────────
# Parquet processing
# ─────────────────────────────────────────────────────────────────────────────

def _determine_bbox_format(table) -> str:
    """
    Auto-detect whether bboxes are xyxy [x1,y1,x2,y2] or xywh [x,y,w,h].

    Strategy: compute area from bbox under each assumption, compare to
    the 'area' field stored in the dataset.  The format with better
    agreement (mean absolute relative error < 30 %) wins.
    Returns 'xyxy' or 'xywh'.
    """
    import numpy as np

    df = table.to_pandas()
    xyxy_errors, xywh_errors = [], []
    checked = 0

    for _, row in df.iterrows():
        objs = row.get("objects", {})
        if not isinstance(objs, dict):
            continue
        bboxes   = objs.get("bbox", [])
        areas    = objs.get("area", [])
        img_w    = row.get("width",  640)
        img_h    = row.get("height", 640)

        for bbox, area in zip(bboxes, areas, strict=False):
            if len(bbox) != 4 or area <= 0:
                continue
            x0, y0, x1, y1 = bbox

            # xyxy interpretation
            area_xyxy = max(0, (x1 - x0)) * max(0, (y1 - y0))
            # xywh interpretation
            area_xywh = x1 * y1  # x1 is w, y1 is h

            if area_xyxy > 0:
                xyxy_errors.append(abs(area_xyxy - area) / area)
            if area_xywh > 0:
                xywh_errors.append(abs(area_xywh - area) / area)

            checked += 1
            if checked >= 200:
                break
        if checked >= 200:
            break

    mean_xyxy = float(np.mean(xyxy_errors)) if xyxy_errors else 1.0
    mean_xywh = float(np.mean(xywh_errors)) if xywh_errors else 1.0

    fmt = "xyxy" if mean_xyxy <= mean_xywh else "xywh"
    print(f"  Bbox format detection: xyxy_err={mean_xyxy:.3f}  xywh_err={mean_xywh:.3f}  → {fmt}")
    return fmt


def _bbox_to_yolo(bbox: list[float], img_w: int, img_h: int, fmt: str) -> list[float] | None:
    """
    Convert a bbox to YOLO (cx, cy, w, h) all normalised to [0, 1].
    Returns None if the box is degenerate.
    """
    if fmt == "xyxy":
        x1, y1, x2, y2 = bbox
        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
    else:  # xywh
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

    # Clamp to image bounds
    cx = max(0.0, min(1.0, cx / img_w))
    cy = max(0.0, min(1.0, cy / img_h))
    w  = max(0.0, min(1.0, w  / img_w))
    h  = max(0.0, min(1.0, h  / img_h))

    if w < 0.005 or h < 0.005:   # discard tiny boxes (< 0.5 % of image)
        return None
    return [cx, cy, w, h]


def process_parquet(
    parquet_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    split: str,
    max_images: int | None = None,
    bbox_fmt: str | None = None,
) -> tuple[dict[str, int], int]:
    """
    Read one parquet shard and write YOLO-format images + labels.

    Returns (instance_counts, n_images_written).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split} parquet: {parquet_path.name}")
    table = pq.read_table(parquet_path)
    df    = table.to_pandas()
    print(f"  Rows in shard: {len(df)}")

    if bbox_fmt is None:
        bbox_fmt = _determine_bbox_format(table)

    instance_counts: dict[str, int] = {n: 0 for n in APP_CLASS_NAMES}
    n_written     = 0
    n_skipped     = 0
    n_no_garment  = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
        if max_images is not None and n_written >= max_images:
            break

        # ── Extract image ──────────────────────────────────────────────────
        img_field = row.get("image")
        if img_field is None:
            n_skipped += 1
            continue

        img_bytes = img_field.get("bytes") if isinstance(img_field, dict) else None
        if not img_bytes:
            n_skipped += 1
            continue

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            n_skipped += 1
            continue

        img_w = int(row.get("width",  pil_img.width))
        img_h = int(row.get("height", pil_img.height))

        # ── Extract and remap bboxes ───────────────────────────────────────
        objs = row.get("objects", {})
        if not isinstance(objs, dict):
            n_no_garment += 1
            continue

        raw_bboxes   = objs.get("bbox",     [])
        raw_cats     = objs.get("category", [])
        yolo_lines   : list[str] = []

        for bbox, src_cat in zip(raw_bboxes, raw_cats, strict=False):
            app_idx = FASHIONPEDIA_TO_APP.get(int(src_cat), -1)
            if app_idx < 0:
                continue  # discard accessories / fine-grained attributes

            yolo_box = _bbox_to_yolo(bbox, img_w, img_h, bbox_fmt)
            if yolo_box is None:
                continue

            cx, cy, w, h = yolo_box
            yolo_lines.append(f"{app_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            instance_counts[APP_CLASS_NAMES[app_idx]] += 1

        if not yolo_lines:
            n_no_garment += 1
            continue  # image has no relevant garment annotations

        # ── Write image ────────────────────────────────────────────────────
        stem     = f"{split}_{idx:06d}"
        img_path = out_img_dir / f"{stem}.jpg"
        if not img_path.exists():
            pil_img.save(img_path, "JPEG", quality=90)

        # ── Write label ────────────────────────────────────────────────────
        lbl_path = out_lbl_dir / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines) + "\n")

        n_written += 1

    print(f"  Written : {n_written} images")
    print(f"  Skipped : {n_skipped} (bad image bytes)")
    print(f"  Dropped : {n_no_garment} (no garment annotations after filtering)")

    return instance_counts, n_written


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(out_root: Path) -> bool:
    """
    Cross-check images ↔ labels; verify class indices; check for
    corrupted images.  Returns True if dataset passes all checks.
    """
    print("\n" + "="*60)
    print("  Dataset Validation")
    print("="*60)

    all_ok = True

    for split in ("train", "val"):
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split

        if not img_dir.exists():
            print(f"  [WARN] Missing: {img_dir}")
            continue

        imgs = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        lbls = {p.stem: p for p in sorted(lbl_dir.iterdir()) if p.suffix == ".txt"}

        # Orphan check
        no_label = [s for s in imgs if s not in lbls]
        no_image = [s for s in lbls if s not in imgs]
        if no_label:
            print(f"  [{split}] {len(no_label)} images have no label file → removing")
            for s in no_label:
                imgs[s].unlink()
            all_ok = False
        if no_image:
            print(f"  [{split}] {len(no_image)} labels have no image file → removing")
            for s in no_image:
                lbls[s].unlink()
            all_ok = False

        # Class index + image integrity check
        bad_class = 0
        bad_img   = 0
        n_inst    = 0
        for stem, lbl_path in tqdm(lbls.items(), desc=f"  Validating {split}", leave=False):
            # Check label
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                if cls_idx < 0 or cls_idx >= 13:
                    bad_class += 1
                n_inst += 1

            # Check image (quick PIL open)
            img_path = imgs.get(stem)
            if img_path:
                try:
                    Image.open(img_path).verify()
                except Exception:
                    bad_img += 1
                    img_path.unlink()
                    lbl_path.unlink()

        n_img = len(list(img_dir.iterdir()))
        print(f"  [{split}] {n_img} images  {n_inst} instances  bad_class={bad_class}  bad_img={bad_img}")
        if bad_class or bad_img:
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset summary
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(out_root: Path) -> None:
    """Print image counts, label counts, class distribution."""
    print("\n" + "="*60)
    print("  Dataset Summary")
    print("="*60)

    total_counts: dict[str, int] = {n: 0 for n in APP_CLASS_NAMES}
    for split in ("train", "val"):
        lbl_dir = out_root / "labels" / split
        img_dir = out_root / "images" / split
        if not lbl_dir.exists():
            continue
        n_img = len(list((img_dir).iterdir())) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.iterdir()))
        split_counts: dict[str, int] = {n: 0 for n in APP_CLASS_NAMES}
        for lbl_path in lbl_dir.iterdir():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    idx = int(parts[0])
                    if 0 <= idx < 13:
                        split_counts[APP_CLASS_NAMES[idx]] += 1
                        total_counts[APP_CLASS_NAMES[idx]] += 1
        print(f"\n  Split: {split}")
        print(f"    Images : {n_img}")
        print(f"    Labels : {n_lbl}")
        total = sum(split_counts.values())
        print(f"    Instances: {total}")
        for cls, cnt in sorted(split_counts.items(), key=lambda x: -x[1]):
            if cnt:
                bar = "█" * min(30, cnt // max(1, total // 30))
                print(f"      {cls:12s} {cnt:6d}  {bar}")

    print("\n  Total class distribution:")
    grand_total = sum(total_counts.values())
    for cls, cnt in sorted(total_counts.items(), key=lambda x: -x[1]):
        if cnt:
            pct = 100 * cnt / grand_total if grand_total else 0
            bar = "█" * min(30, cnt // max(1, grand_total // 30))
            print(f"    {cls:12s} {cnt:6d}  {pct:5.1f}%  {bar}")

    n_classes_present = sum(1 for v in total_counts.values() if v > 0)
    print(f"\n  Classes present : {n_classes_present} / 13")
    print(f"  Total instances : {grand_total}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Fashionpedia and convert to app YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "data" / "fashion_dataset",
        help="Output dataset root (default: data/fashion_dataset).",
    )
    parser.add_argument(
        "--train-shards", type=int, default=1,
        help="Number of train parquet shards to download (1-8, default 1 ≈ 5 700 images).",
    )
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Cap total training images (overrides --train-shards if smaller).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data" / "_fashionpedia_cache",
        help="Directory to cache downloaded parquet files.",
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Keep parquet cache after processing (avoids re-download).",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip post-processing validation step.",
    )
    args = parser.parse_args()

    out_root  : Path = args.out_root.resolve()
    cache_dir : Path = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_shards = max(1, min(8, args.train_shards))
    train_urls = TRAIN_SHARD_URLS[:n_shards]

    print("\n" + "="*60)
    print("  Fashionpedia Dataset Preparation")
    print("="*60)
    print(f"  Output root  : {out_root}")
    print(f"  Train shards : {n_shards} (~{n_shards * 5700} raw images)")
    print("  Val shard    : 1 (~1 158 images)")
    print(f"  Cache dir    : {cache_dir}")

    # ── Clear existing synthetic dataset ──────────────────────────────────────
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out_root / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    print(f"\n  Cleared existing dataset at {out_root}")

    # ── Download parquet shards ───────────────────────────────────────────────
    train_parquets: list[Path] = []
    for i, url in enumerate(train_urls):
        dest = cache_dir / f"train_{i:04d}.parquet"
        if dest.exists() and dest.stat().st_size > 10_000_000:
            print(f"  [cache] train shard {i}: {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        else:
            print(f"\n  Downloading train shard {i} (~480 MB) …")
            download_file(url, dest, desc=f"train shard {i}")
        train_parquets.append(dest)

    val_parquet = cache_dir / "val_0000.parquet"
    if val_parquet.exists() and val_parquet.stat().st_size > 1_000_000:
        print(f"  [cache] val shard: {val_parquet.name} ({val_parquet.stat().st_size/1e6:.0f} MB)")
    else:
        print("\n  Downloading val shard (~85 MB) …")
        download_file(VAL_SHARD_URL, val_parquet, desc="val shard")

    # ── Process shards → YOLO format ─────────────────────────────────────────
    # Determine bbox format from first train shard (shared across all shards)
    print("\n  Detecting bbox format from shard 0 …")
    tbl0 = pq.read_table(train_parquets[0])
    bbox_fmt = _determine_bbox_format(tbl0)
    del tbl0

    all_counts: dict[str, int] = {n: 0 for n in APP_CLASS_NAMES}

    # Train
    max_per_shard = (
        (args.max_train + n_shards - 1) // n_shards
        if args.max_train else None
    )
    for i, pq_path in enumerate(train_parquets):
        counts, _ = process_parquet(
            pq_path,
            out_root / "images" / "train",
            out_root / "labels" / "train",
            split="train",
            max_images=max_per_shard,
            bbox_fmt=bbox_fmt,
        )
        for k, v in counts.items():
            all_counts[k] += v

    # Val
    val_counts, _ = process_parquet(
        val_parquet,
        out_root / "images" / "val",
        out_root / "labels" / "val",
        split="val",
        max_images=None,
        bbox_fmt=bbox_fmt,
    )
    for k, v in val_counts.items():
        all_counts[k] += v

    # ── Clean up cache ────────────────────────────────────────────────────────
    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("\n  Parquet cache cleared.")

    # ── Validate dataset ──────────────────────────────────────────────────────
    if not args.no_validate:
        ok = validate_dataset(out_root)
        status = "✓ PASSED" if ok else "⚠ WARNINGS (see above)"
        print(f"\n  Validation: {status}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_dataset_summary(out_root)
    print(f"\n✓ Dataset ready at: {out_root}")
    print("  Next step: python3.11 scripts/train_fashion_yolo.py --epochs 100")


if __name__ == "__main__":
    main()
