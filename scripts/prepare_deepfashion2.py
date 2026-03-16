#!/usr/bin/env python3
"""
scripts/prepare_deepfashion2.py
================================
Convert a DeepFashion2 split into YOLO-format labels for fine-tuning.

DeepFashion2
────────────
DeepFashion2 is a comprehensive fashion benchmark with 491K images and 801K
clothing instances across 13 fine-grained categories.

Download instructions
─────────────────────
1. Register at https://github.com/switchablenorms/DeepFashion2
2. Request access and download the dataset (≈25 GB for the full set).
3. Extract to a directory and pass it to --df2-root.

Expected DeepFashion2 directory layout::

    <df2_root>/
    ├── train/
    │   ├── image/        ← JPEG images (01234.jpg …)
    │   └── annots/       ← JSON annotations (01234.json …)
    └── validation/
        ├── image/
        └── annots/

Each per-image JSON has the structure::

    {
      "item1": {
        "category_id": 1,           # 1-indexed (1–13)
        "category_name": "short sleeve top",
        "bounding_box": [x1, y1, x2, y2],   # pixel coords, 0-indexed
        "landmarks": [...],
        ...
      },
      "item2": { ... },
      ...
    }

Output layout
─────────────
The script produces::

    data/fashion_dataset/
    ├── images/
    │   ├── train/    ← copies (or symlinks) of the original images
    │   └── val/
    └── labels/
        ├── train/    ← YOLO .txt files (one row per garment)
        └── val/

Usage
─────
    python scripts/prepare_deepfashion2.py \\
        --df2-root  /path/to/deepfashion2 \\
        --out-root  data/fashion_dataset \\
        --val-frac  0.1 \\
        --symlink           # create symlinks instead of copying images
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ── Repo root on PYTHONPATH ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.fashion_classes import (
    APP_CLASS_NAMES,
    DEEPFASHION2_CLASSES,
    name_to_category,
)

# ── DeepFashion2 category_id (1-indexed) → app class index (0-indexed) ───────
# DEEPFASHION2_CLASSES is keyed 0-12; DF2 JSON uses 1-13.
_DF2_CAT_TO_APP_IDX: dict[int, int] = {}
for _app_idx, (_name, _cat) in DEEPFASHION2_CLASSES.items():
    # Find matching app class index by category name
    for _j, _app_name in enumerate(APP_CLASS_NAMES):
        if name_to_category(_app_name) == _cat and _j not in _DF2_CAT_TO_APP_IDX.values():
            _DF2_CAT_TO_APP_IDX[_app_idx + 1] = _j  # DF2 is 1-indexed
            break

# Fallback: map remaining DF2 categories directly via name
for _df2_idx, (_df2_name, _df2_cat) in DEEPFASHION2_CLASSES.items():
    _df2_key = _df2_idx + 1
    if _df2_key not in _DF2_CAT_TO_APP_IDX:
        for _j, _app_name in enumerate(APP_CLASS_NAMES):
            if name_to_category(_app_name) == _df2_cat:
                _DF2_CAT_TO_APP_IDX[_df2_key] = _j
                break


def convert_split(
    image_dir: Path,
    annot_dir: Path,
    out_images: Path,
    out_labels: Path,
    symlink: bool = False,
    max_samples: int | None = None,
) -> int:
    """
    Convert one DeepFashion2 split (train or validation) to YOLO format.

    Returns the number of images successfully processed.
    """
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annot_dir.glob("*.json"))
    if max_samples:
        json_files = json_files[:max_samples]

    processed = 0
    skipped   = 0

    for annot_path in json_files:
        img_path = image_dir / (annot_path.stem + ".jpg")
        if not img_path.exists():
            img_path = image_dir / (annot_path.stem + ".jpeg")
        if not img_path.exists():
            skipped += 1
            continue

        # ── Load annotation ──────────────────────────────────────────────────
        try:
            with open(annot_path) as f:
                annot = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        # Get image dimensions from the first item (all share the same image)
        img_width  = annot.get("width",  None)
        img_height = annot.get("height", None)

        # DeepFashion2 stores dimensions as top-level keys in some splits
        if img_width is None or img_height is None:
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil:
                    img_width, img_height = pil.size
            except Exception:
                skipped += 1
                continue

        yolo_rows: list[str] = []

        for item_key, item in annot.items():
            if not item_key.startswith("item"):
                continue  # skip top-level width/height keys

            cat_id = item.get("category_id")
            bbox   = item.get("bounding_box")

            if cat_id is None or bbox is None:
                continue

            # Map DF2 category_id → app class index
            app_class_idx = _DF2_CAT_TO_APP_IDX.get(int(cat_id))
            if app_class_idx is None:
                continue  # unknown category

            x1, y1, x2, y2 = bbox
            # Clamp to image bounds
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = min(float(img_width),  float(x2))
            y2 = min(float(img_height), float(y2))

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            # Convert to YOLO normalised format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width    = (x2 - x1) / img_width
            height   = (y2 - y1) / img_height

            yolo_rows.append(
                f"{app_class_idx} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )

        if not yolo_rows:
            skipped += 1
            continue

        # ── Write YOLO label file ─────────────────────────────────────────────
        label_path = out_labels / (annot_path.stem + ".txt")
        label_path.write_text("\n".join(yolo_rows) + "\n")

        # ── Copy / symlink image ──────────────────────────────────────────────
        dest_img = out_images / img_path.name
        if not dest_img.exists():
            if symlink:
                dest_img.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dest_img)

        processed += 1

    print(f"  processed={processed}  skipped={skipped}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 to YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--df2-root",
        type=Path,
        required=True,
        help="Root directory of the extracted DeepFashion2 dataset.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "fashion_dataset",
        help="Output directory (default: data/fashion_dataset).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of training images to hold out as validation (default: 0.1). "
             "If a 'validation' split exists in df2-root it is used directly.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying images (saves disk space).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit conversion to this many images per split (useful for quick tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    args = parser.parse_args()

    df2_root: Path = args.df2_root.resolve()
    out_root: Path = args.out_root.resolve()

    if not df2_root.exists():
        sys.exit(f"[ERROR] df2-root not found: {df2_root}")

    random.seed(args.seed)

    print(f"DeepFashion2 root : {df2_root}")
    print(f"Output root       : {out_root}")
    print()

    # ── Category mapping summary ──────────────────────────────────────────────
    print("DeepFashion2 → app class mapping:")
    for df2_idx, app_idx in sorted(_DF2_CAT_TO_APP_IDX.items()):
        df2_name = DEEPFASHION2_CLASSES[df2_idx - 1][0]
        app_name = APP_CLASS_NAMES[app_idx]
        print(f"  DF2 {df2_idx:2d} '{df2_name}' → app {app_idx} '{app_name}'")
    print()

    # ── Determine source splits ───────────────────────────────────────────────
    train_img_dir  = df2_root / "train"  / "image"
    train_ann_dir  = df2_root / "train"  / "annots"
    val_img_dir    = df2_root / "validation" / "image"
    val_ann_dir    = df2_root / "validation" / "annots"

    has_val_split = val_img_dir.exists() and val_ann_dir.exists()

    if has_val_split:
        print("Using official train + validation splits.")
        print("\n[train]")
        convert_split(
            train_img_dir, train_ann_dir,
            out_root / "images" / "train",
            out_root / "labels" / "train",
            symlink=args.symlink,
            max_samples=args.max_samples,
        )
        print("\n[val]")
        convert_split(
            val_img_dir, val_ann_dir,
            out_root / "images" / "val",
            out_root / "labels" / "val",
            symlink=args.symlink,
            max_samples=args.max_samples,
        )
    else:
        # No dedicated val split: carve one from train
        print(
            f"No 'validation/' split found — creating {args.val_frac:.0%} val "
            f"split from training data."
        )
        all_json = sorted(train_ann_dir.glob("*.json"))
        random.shuffle(all_json)
        n_val = max(1, int(len(all_json) * args.val_frac))
        val_stems   = {p.stem for p in all_json[:n_val]}
        train_stems = {p.stem for p in all_json[n_val:]}

        def _filtered_convert(
            stems: set[str], split_name: str
        ) -> None:
            img_out = out_root / "images" / split_name
            lbl_out = out_root / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            # Temporarily create a filtered annots dir in memory
            processed = 0
            for ann_path in sorted(train_ann_dir.glob("*.json")):
                if ann_path.stem not in stems:
                    continue
                img_path = train_img_dir / (ann_path.stem + ".jpg")
                if not img_path.exists():
                    continue
                try:
                    with open(ann_path) as f:
                        annot = json.load(f)
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as pil:
                        img_width, img_height = pil.size
                except Exception:
                    continue

                yolo_rows = []
                for item_key, item in annot.items():
                    if not item_key.startswith("item"):
                        continue
                    cat_id = item.get("category_id")
                    bbox   = item.get("bounding_box")
                    if cat_id is None or bbox is None:
                        continue
                    app_class_idx = _DF2_CAT_TO_APP_IDX.get(int(cat_id))
                    if app_class_idx is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                    x2, y2 = min(float(img_width), float(x2)), min(float(img_height), float(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    xc = ((x1 + x2) / 2) / img_width
                    yc = ((y1 + y2) / 2) / img_height
                    w  = (x2 - x1) / img_width
                    h  = (y2 - y1) / img_height
                    yolo_rows.append(f"{app_class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                if not yolo_rows:
                    continue
                (lbl_out / (ann_path.stem + ".txt")).write_text("\n".join(yolo_rows) + "\n")
                dest_img = img_out / img_path.name
                if not dest_img.exists():
                    if args.symlink:
                        dest_img.symlink_to(img_path.resolve())
                    else:
                        shutil.copy2(img_path, dest_img)
                processed += 1
            print(f"  {split_name}: {processed} images")

        print("\n[train]")
        _filtered_convert(train_stems, "train")
        print("\n[val]")
        _filtered_convert(val_stems, "val")

    print(f"\n✓ Dataset written to {out_root}")
    print("  Next step: python scripts/train_fashion_yolo.py")


if __name__ == "__main__":
    main()
