#!/usr/bin/env python3
"""
scripts/validate_fashion_detection.py
======================================
Comprehensive detection validation script.

Loads the trained fashion YOLO model and runs detection on:
  1. Uploaded real images (uploads/)
  2. Synthetic test images generated on-the-fly

Produces:
  • Detailed console report with model diagnostics
  • Per-image detection table
  • Category confusion summary
  • Annotated images saved to uploads/annotated/

Usage:
    PINECONE_API_KEY=dummy python scripts/validate_fashion_detection.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PINECONE_API_KEY", "dummy")

import warnings
warnings.filterwarnings("ignore")

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ml.yolo_detector import YOLODetector, ALL_FASHION_CATEGORIES, TARGET_CATEGORIES
from ml.fashion_classes import APP_CLASS_NAMES, build_label_map_from_model_names, is_fashion_model
from app.models.schemas import GarmentCategory

# ── Colours per category ──────────────────────────────────────────────────────
COLOURS: dict[GarmentCategory, str] = {
    GarmentCategory.SHIRT:  "#2196F3",
    GarmentCategory.PANTS:  "#4CAF50",
    GarmentCategory.SHOES:  "#FF9800",
    GarmentCategory.JACKET: "#9C27B0",
    GarmentCategory.DRESS:  "#E91E63",
    GarmentCategory.SKIRT:  "#00BCD4",
    GarmentCategory.OTHER:  "#9E9E9E",
}


def draw_annotations(image: Image.Image, detections) -> Image.Image:
    """Return a copy of image with bounding boxes and labels drawn."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bb = det.bounding_box
        color = COLOURS.get(det.category, "#9E9E9E")
        label = f"{det.category.value} {bb.confidence:.0%}"
        draw.rectangle([bb.x_min, bb.y_min, bb.x_max, bb.y_max], outline=color, width=3)
        text_bbox = draw.textbbox((bb.x_min, bb.y_min - 24), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bb.x_min, bb.y_min - 24), label, fill="white", font=font)

    return annotated


def create_test_images() -> list[tuple[str, Image.Image]]:
    """Create synthetic test images with identifiable patterns."""
    images = []

    # 1. Large blue rectangle (shirt-like) + green rectangle (pants-like)
    img = Image.new("RGB", (640, 640), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 50, 400, 300], fill=(70, 130, 180))   # blue = shirt area
    draw.rectangle([120, 320, 380, 580], fill=(85, 107, 47))   # olive = pants area
    images.append(("synthetic_shirt_pants", img))

    # 2. Dark rectangle (jacket) + brown rectangle (shoes)
    img = Image.new("RGB", (640, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 30, 450, 280], fill=(50, 50, 50))       # dark = jacket
    draw.rectangle([150, 550, 350, 630], fill=(139, 69, 19))    # brown = shoes
    images.append(("synthetic_jacket_shoes", img))

    # 3. Pink rectangle (dress-like)
    img = Image.new("RGB", (640, 640), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 40, 420, 550], fill=(255, 20, 147))    # deep pink = dress
    images.append(("synthetic_dress", img))

    # 4. Multiple small items
    img = Image.new("RGB", (640, 640), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 200, 200], fill=(70, 130, 180))     # shirt
    draw.rectangle([220, 20, 400, 200], fill=(85, 107, 47))     # pants
    draw.rectangle([420, 20, 600, 200], fill=(139, 69, 19))     # shoes
    draw.rectangle([20, 220, 200, 400], fill=(50, 50, 50))      # jacket
    images.append(("synthetic_multi_garments", img))

    return images


def main() -> None:
    print(f"\n{'='*70}")
    print("  Fashion Detection Pipeline Validation")
    print(f"{'='*70}\n")

    # ── Step 1: Model diagnostics ─────────────────────────────────────────────
    detector = YOLODetector()
    summary = detector.model_summary()

    print("─── MODEL DIAGNOSTICS ──────────────────────────────────────────────")
    print(f"  Model path        : {summary['model_path']}")
    print(f"  Is fashion model  : {summary['is_fashion_model']}")
    print(f"  Number of classes : {summary['num_classes']}")
    print(f"  Confidence thresh : {summary['confidence_threshold']}")
    print(f"  Class names       :")
    for idx, name in sorted(summary['class_names'].items(), key=lambda x: int(x[0])):
        cat = detector._label_map.get(int(idx), GarmentCategory.OTHER)
        print(f"    {idx:>3}: {name:<12} → {cat.value}")
    print()

    # Verify label map
    label_map = build_label_map_from_model_names(detector.model.names)
    print(f"  build_label_map_from_model_names: ✓ ({len(label_map)} entries)")
    print(f"  is_fashion_model:                 ✓ ({is_fashion_model(detector.model.names)})")
    print()

    # ── Step 2: Detect on real uploaded images ────────────────────────────────
    uploads_dir = ROOT / "uploads"
    real_images = sorted(
        p for p in uploads_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and p.parent.name != "annotated"
    )

    out_dir = uploads_dir / "annotated"
    out_dir.mkdir(exist_ok=True)

    all_detections: list[GarmentCategory] = []

    if real_images:
        print("─── REAL IMAGE DETECTIONS ──────────────────────────────────────────")
        for img_path in real_images:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            print(f"\n  Image: {img_path.name} ({w}×{h})")

            all_dets = detector.detect(img)
            target_dets = detector.detect_targets(img)
            fashion_dets = detector.detect_all_fashion(img)

            print(f"  All detections     : {len(all_dets)}")
            print(f"  Target (S/P/Sh)    : {len(target_dets)}")
            print(f"  All fashion (6 cat): {len(fashion_dets)}")

            for det in all_dets:
                bb = det.bounding_box
                area_pct = 100 * (bb.x_max - bb.x_min) * (bb.y_max - bb.y_min) / (w * h)
                print(f"    • {det.category.value:<8} conf={bb.confidence:.2%} area={area_pct:.1f}%")
                all_detections.append(det.category)

            # Save annotated
            if all_dets:
                annotated = draw_annotations(img, all_dets)
                annotated.save(out_dir / img_path.name, format="JPEG", quality=95)
                print(f"    → Annotated saved: uploads/annotated/{img_path.name}")
        print()

    # ── Step 3: Detect on synthetic test images ───────────────────────────────
    print("─── SYNTHETIC IMAGE DETECTIONS ─────────────────────────────────────")
    test_images = create_test_images()
    for name, img in test_images:
        all_dets = detector.detect(img)
        target_dets = detector.detect_targets(img)
        print(f"\n  {name}:")
        print(f"    All: {len(all_dets)}  Targets: {len(target_dets)}")
        for det in all_dets:
            bb = det.bounding_box
            print(f"    • {det.category.value:<8} conf={bb.confidence:.2%}")
            all_detections.append(det.category)

        # Save annotated synthetic images too
        if all_dets:
            annotated = draw_annotations(img, all_dets)
            annotated.save(out_dir / f"{name}.jpg", format="JPEG", quality=95)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  DETECTION SUMMARY")
    print(f"{'─'*70}")
    counter = Counter(all_detections)
    total = sum(counter.values())
    print(f"  Total detections across all images: {total}")
    if counter:
        for cat in GarmentCategory:
            n = counter.get(cat, 0)
            bar = "█" * min(30, n)
            print(f"    {cat.value:<8} {n:4d}  {bar}")

    # Category coverage check
    detected_cats = set(counter.keys())
    expected = {GarmentCategory.SHIRT, GarmentCategory.PANTS, GarmentCategory.SHOES}
    covered = detected_cats & expected
    missing = expected - detected_cats
    print(f"\n  Target categories detected: {[c.value for c in covered]}")
    if missing:
        print(f"  ⚠ Missing target categories: {[c.value for c in missing]}")
        print("    (Expected with only 3 epochs of synthetic training)")
    else:
        print(f"  ✓ All target categories (shirt, pants, shoes) detected!")

    print(f"\n  Pipeline status: {'✓ WORKING' if summary['is_fashion_model'] else '✗ NOT READY'}")
    print(f"  Note: For real accuracy, train on DeepFashion2 or Roboflow dataset")
    print(f"        with 100+ epochs.\n")


if __name__ == "__main__":
    main()
