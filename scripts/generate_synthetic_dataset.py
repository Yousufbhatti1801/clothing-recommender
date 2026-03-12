#!/usr/bin/env python3
"""
scripts/generate_synthetic_dataset.py
======================================
Generate a small synthetic fashion dataset for smoke-testing the training
pipeline.  Produces coloured rectangles on random backgrounds with correct
YOLO-format annotations.

This is NOT a real fashion dataset — it exists solely to validate that:
  1. config/fashion_dataset.yaml is correct
  2. train_fashion_yolo.py runs without errors
  3. The trained model's class names match APP_CLASS_NAMES

For actual fine-tuning, use prepare_deepfashion2.py or prepare_fashion_roboflow.py.

Usage:
    python scripts/generate_synthetic_dataset.py              # default 200 images
    python scripts/generate_synthetic_dataset.py --n-train 50 --n-val 10
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from ml.fashion_classes import APP_CLASS_NAMES

# Distinct colours per class (used to tint fake "garment" rectangles)
CLASS_COLORS: list[tuple[int, int, int]] = [
    (70, 130, 180),   # shirt — steel blue
    (100, 149, 237),  # t-shirt — cornflower blue
    (85, 107, 47),    # pants — dark olive green
    (0, 0, 139),      # jeans — dark blue
    (154, 205, 50),   # shorts — yellow green
    (139, 69, 19),    # shoes — saddle brown
    (255, 255, 255),  # sneakers — white
    (50, 50, 50),     # jacket — dark grey
    (105, 105, 105),  # coat — dim grey
    (255, 20, 147),   # dress — deep pink
    (255, 105, 180),  # skirt — hot pink
    (210, 180, 140),  # bag — tan
    (255, 215, 0),    # hat — gold
]


def random_bg_color() -> tuple[int, int, int]:
    return (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))


def generate_image(
    img_size: int = 640,
    min_objects: int = 1,
    max_objects: int = 4,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic image with random coloured rectangles.

    Returns:
        (PIL Image, list of YOLO annotations: [(cls_idx, xc, yc, w, h), ...])
    """
    img = Image.new("RGB", (img_size, img_size), color=random_bg_color())
    draw = ImageDraw.Draw(img)

    n_objects = random.randint(min_objects, max_objects)
    annotations: list[tuple[int, float, float, float, float]] = []

    for _ in range(n_objects):
        cls_idx = random.randint(0, len(APP_CLASS_NAMES) - 1)
        color = CLASS_COLORS[cls_idx]

        # Random box (ensuring minimum 40px in each dimension)
        w = random.randint(60, img_size // 2)
        h = random.randint(60, img_size // 2)
        x1 = random.randint(0, img_size - w)
        y1 = random.randint(0, img_size - h)
        x2 = x1 + w
        y2 = y1 + h

        # Draw filled rectangle with slight variation
        r, g, b = color
        varied = (
            max(0, min(255, r + random.randint(-20, 20))),
            max(0, min(255, g + random.randint(-20, 20))),
            max(0, min(255, b + random.randint(-20, 20))),
        )
        draw.rectangle([x1, y1, x2, y2], fill=varied, outline=(0, 0, 0), width=2)

        # Add some texture (diagonal lines)
        for offset in range(0, max(w, h), 8):
            draw.line(
                [(x1 + offset, y1), (x1, y1 + offset)],
                fill=(varied[0] // 2, varied[1] // 2, varied[2] // 2),
                width=1,
            )

        # YOLO normalised annotation
        xc = ((x1 + x2) / 2) / img_size
        yc = ((y1 + y2) / 2) / img_size
        nw = (x2 - x1) / img_size
        nh = (y2 - y1) / img_size
        annotations.append((cls_idx, xc, yc, nw, nh))

    return img, annotations


def generate_split(
    out_img_dir: Path,
    out_lbl_dir: Path,
    n_images: int,
    img_size: int = 640,
) -> dict[str, int]:
    """Generate images and labels for one split. Returns per-class counts."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {name: 0 for name in APP_CLASS_NAMES}

    for i in range(n_images):
        img, annotations = generate_image(img_size=img_size)
        stem = f"{i:06d}"

        # Save image
        img.save(out_img_dir / f"{stem}.jpg", format="JPEG", quality=90)

        # Save YOLO label
        lines = []
        for cls_idx, xc, yc, w, h in annotations:
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[APP_CLASS_NAMES[cls_idx]] += 1

        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fashion dataset for pipeline testing.",
    )
    parser.add_argument("--n-train", type=int, default=200, help="Training images (default 200)")
    parser.add_argument("--n-val",   type=int, default=40,  help="Validation images (default 40)")
    parser.add_argument("--img-size", type=int, default=640, help="Image size (default 640)")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "fashion_dataset",
        help="Output directory (default: data/fashion_dataset)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    out_root: Path = args.out_root

    print(f"Generating synthetic fashion dataset ({args.n_train} train, {args.n_val} val)")
    print(f"Output: {out_root}")
    print(f"Classes: {len(APP_CLASS_NAMES)}: {APP_CLASS_NAMES}\n")

    print("[train]")
    train_counts = generate_split(
        out_root / "images" / "train",
        out_root / "labels" / "train",
        args.n_train,
        args.img_size,
    )

    print("[val]")
    val_counts = generate_split(
        out_root / "images" / "val",
        out_root / "labels" / "val",
        args.n_val,
        args.img_size,
    )

    print("\nPer-class instance counts:")
    print(f"  {'class':<12} {'train':>6} {'val':>6}")
    print(f"  {'─'*12} {'─'*6} {'─'*6}")
    for name in APP_CLASS_NAMES:
        print(f"  {name:<12} {train_counts[name]:>6} {val_counts[name]:>6}")

    total_train = sum(train_counts.values())
    total_val   = sum(val_counts.values())
    print(f"  {'TOTAL':<12} {total_train:>6} {total_val:>6}")

    print(f"\n✓ Synthetic dataset ready at {out_root}")
    print("  Next step: python scripts/train_fashion_yolo.py --epochs 3 --batch 4")


if __name__ == "__main__":
    main()
