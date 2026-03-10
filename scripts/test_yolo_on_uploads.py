"""
scripts/test_yolo_on_uploads.py
================================
Runs YOLOv8 clothing detection on every image inside uploads/ and produces:
  • A console report per image (category, confidence, bounding box, area %)
  • Annotated images saved to uploads/annotated/ with boxes + labels drawn

Usage:
    python scripts/test_yolo_on_uploads.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from ml.yolo_detector import LABEL_MAP, TARGET_CATEGORIES, YOLODetector
from app.models.schemas import GarmentCategory

# ── colour palette per category ──────────────────────────────────────────────
COLOURS: dict[GarmentCategory, str] = {
    GarmentCategory.SHIRT:  "#2196F3",   # blue
    GarmentCategory.PANTS:  "#4CAF50",   # green
    GarmentCategory.SHOES:  "#FF9800",   # orange
    GarmentCategory.JACKET: "#9C27B0",   # purple
    GarmentCategory.DRESS:  "#E91E63",   # pink
    GarmentCategory.SKIRT:  "#00BCD4",   # cyan
    GarmentCategory.OTHER:  "#9E9E9E",   # grey
}

LINE_WIDTH = 3


def draw_annotations(image: Image.Image, detections) -> Image.Image:
    """Return a copy of *image* with bounding boxes and labels drawn."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Try to load a truetype font; fall back to the built-in bitmap font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bb    = det.bounding_box
        color = COLOURS.get(det.category, "#9E9E9E")
        label = f"{det.category.value} {det.bounding_box.confidence:.0%}"

        # Bounding box rectangle
        draw.rectangle(
            [bb.x_min, bb.y_min, bb.x_max, bb.y_max],
            outline=color,
            width=LINE_WIDTH,
        )

        # Label background + text
        text_bbox = draw.textbbox((bb.x_min, bb.y_min - 24), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bb.x_min, bb.y_min - 24), label, fill="white", font=font)

    return annotated


def area_pct(det, img_w: int, img_h: int) -> float:
    bb = det.bounding_box
    box_area = (bb.x_max - bb.x_min) * (bb.y_max - bb.y_min)
    return 100.0 * box_area / (img_w * img_h) if img_w * img_h else 0.0


def report_image(image_path: Path, detector: YOLODetector) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Image : {image_path.name}")
    print(f"{'═' * 60}")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"  Size  : {img_w} × {img_h} px")

    # ── ALL detections (every YOLO class) ────────────────────────────────
    all_dets = detector.detect(image)
    print(f"\n  ┌─ ALL detections ({len(all_dets)} total) ──────────────────")
    if all_dets:
        for i, det in enumerate(all_dets, 1):
            bb   = det.bounding_box
            pct  = area_pct(det, img_w, img_h)
            tiny = " ⚠ TINY (< 1 %)" if pct < 1.0 else ""
            target_mark = "✓" if det.category in TARGET_CATEGORIES else "○"
            print(
                f"  │ {i:>2}. [{target_mark}] {det.category.value:<8}  "
                f"conf={bb.confidence:.2%}  "
                f"box=[{bb.x_min:.0f},{bb.y_min:.0f} → {bb.x_max:.0f},{bb.y_max:.0f}]  "
                f"area={pct:.1f}%{tiny}"
            )
    else:
        print("  │  (no detections above confidence threshold)")
    print("  └────────────────────────────────────────────────────────")

    # ── TARGET-ONLY detections (shirt / pants / shoes, non-tiny) ─────────
    target_dets = [
        d for d in all_dets
        if d.category in TARGET_CATEGORIES and area_pct(d, img_w, img_h) >= 1.0
    ]
    print(f"\n  ┌─ TARGET detections filtered (shirt/pants/shoes, area ≥ 1 %) ─")
    if target_dets:
        for det in target_dets:
            bb  = det.bounding_box
            print(
                f"  │  • {det.category.value:<8}  conf={bb.confidence:.2%}  "
                f"w={bb.x_max - bb.x_min:.0f} h={bb.y_max - bb.y_min:.0f}"
            )
    else:
        print("  │  (none — image may not contain recognisable clothing)")
    print("  └────────────────────────────────────────────────────────")

    # ── Category summary ─────────────────────────────────────────────────
    by_cat: dict[GarmentCategory, int] = {}
    for d in all_dets:
        by_cat[d.category] = by_cat.get(d.category, 0) + 1
    if by_cat:
        print(f"\n  Category summary: " + ", ".join(
            f"{cat.value}×{n}" for cat, n in sorted(by_cat.items(), key=lambda x: x[0])
        ))

    # ── Save annotated image ──────────────────────────────────────────────
    out_dir = image_path.parent / "annotated"
    out_dir.mkdir(exist_ok=True)
    annotated_path = out_dir / image_path.name

    if all_dets:
        annotated = draw_annotations(image, all_dets)
        annotated.save(annotated_path, format="JPEG", quality=95)
        print(f"\n  Annotated image saved → {annotated_path.relative_to(ROOT)}")
    else:
        print(f"\n  No annotations drawn (no detections).")


def main() -> None:
    uploads_dir = ROOT / "uploads"
    images = sorted(
        p for p in uploads_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and p.parent.name != "annotated"
    )

    if not images:
        print("No images found in uploads/. Upload .jpg / .png / .webp files first.")
        return

    print(f"\nLoading YOLOv8 model …")
    detector = YOLODetector()
    print(f"Model loaded.  Confidence threshold = {detector.confidence_threshold}")
    print(f"Label map     : { {k: v.value for k, v in LABEL_MAP.items()} }")
    print(f"Target cats   : {[c.value for c in TARGET_CATEGORIES]}")
    print(f"Images found  : {[p.name for p in images]}")

    for img_path in images:
        report_image(img_path, detector)

    print(f"\n{'═' * 60}")
    print("  Done.  Annotated images saved to uploads/annotated/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
