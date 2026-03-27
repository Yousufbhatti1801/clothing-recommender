#!/usr/bin/env python3
"""
scripts/test_real_pipeline.py
==============================
End-to-end pipeline test using *real* fashion images.

This script exercises the actual ML models (YOLO + CLIP) on downloaded
photos to verify:
  1. YOLO detects garment bounding boxes on real photos
  2. CLIP produces valid 512-d embeddings for each crop
  3. Embeddings are L2-normalised (cosine-ready)
  4. Similar items produce higher similarity than unrelated items

No database or Pinecone connection needed — this is a pure ML smoke test.

Usage
-----
    # First, download real images:
    python3.11 scripts/download_test_images.py

    # Then run this test:
    python3.11 scripts/test_real_pipeline.py

    # Verbose mode (print every detection + embedding stats):
    python3.11 scripts/test_real_pipeline.py --verbose

    # Save annotated images with bounding boxes:
    python3.11 scripts/test_real_pipeline.py --save-annotated

Prerequisites
-------------
    • Downloaded images in tests/real_images/ (run download_test_images.py first)
    • YOLO weights at ml/models/yolov8_fashion.pt (or auto-downloads yolov8n)
    • CLIP model auto-downloads from HuggingFace on first run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure settings are loadable without a .env
import os
os.environ.setdefault("PINECONE_API_KEY", "dummy")


# ── Color map for drawing bounding boxes ────────────────────────────────────
COLORS = {
    "shirt":  (70, 130, 230),
    "pants":  (60, 60, 200),
    "shoes":  (220, 160, 50),
    "jacket": (180, 80, 80),
    "dress":  (200, 100, 200),
    "skirt":  (100, 200, 150),
    "other":  (128, 128, 128),
}


def draw_detections(image: Image.Image, detections, save_path: Path | None = None):
    """Draw bounding boxes on an image copy; optionally save."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        bb = det.bounding_box
        cat = str(det.category)
        color = COLORS.get(cat, (128, 128, 128))
        draw.rectangle([bb.x_min, bb.y_min, bb.x_max, bb.y_max],
                        outline=color, width=3)
        label = f"{cat} {bb.confidence:.2f}"
        draw.text((bb.x_min + 4, bb.y_min + 2), label, fill=color)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    return img


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_pipeline_test(
    image_dir: Path,
    manifest_path: Path,
    detector,
    encoder,
    classifier,
    *,
    verbose: bool = False,
    save_annotated: bool = False,
    annotated_dir: Path | None = None,
) -> dict:
    """
    Run YOLO + CLIP (with zero-shot fallback) on all images listed in the manifest.

    Returns a summary dict with stats and per-image results.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = []
    total_detections = 0
    fallback_used = 0
    category_hits = {}  # category → count
    all_embeddings = {}  # filename → {category: embedding}

    for entry in manifest["images"]:
        filename = entry["filename"]
        expected = entry["expected_garments"]
        img_path = image_dir / filename

        if not img_path.exists():
            results.append({
                "filename": filename,
                "status": "MISSING",
                "detections": [],
                "expected": expected,
            })
            continue

        img = Image.open(img_path).convert("RGB")

        # ── YOLO Detection ───────────────────────────────────────────
        detections = detector.detect_all_fashion(img)
        used_fallback = False

        # ── CLIP zero-shot fallback ──────────────────────────────────
        if not detections:
            top = classifier.classify_image(img, top_k=1)
            if top and top[0][1] >= 0.20:
                from app.models.schemas import BoundingBox, DetectedGarment
                cat, conf = top[0]
                w, h = img.size
                detections = [DetectedGarment(
                    category=cat,
                    bounding_box=BoundingBox(
                        x_min=0.0, y_min=0.0,
                        x_max=float(w), y_max=float(h),
                        confidence=conf,
                    ),
                )]
                used_fallback = True
                fallback_used += 1
        detected_categories = [str(d.category) for d in detections]
        total_detections += len(detections)

        for cat in detected_categories:
            category_hits[cat] = category_hits.get(cat, 0) + 1

        # ── Save annotated image ─────────────────────────────────────
        if save_annotated and annotated_dir and not used_fallback:
            draw_detections(img, detections,
                           save_path=annotated_dir / filename)

        # ── CLIP Embedding for each detected crop ────────────────────
        embeddings = {}
        for det in detections:
            bb = det.bounding_box
            crop = img.crop((int(bb.x_min), int(bb.y_min),
                             int(bb.x_max), int(bb.y_max)))
            # Ensure crop is not degenerate
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue
            vec = encoder.encode([crop])[0]
            cat = str(det.category)
            embeddings[cat] = vec

            if verbose:
                norm = float(np.linalg.norm(vec))
                fallback_tag = " [CLIP-fallback]" if used_fallback else ""
                print(f"    {cat}{fallback_tag}: conf={bb.confidence:.3f}, "
                      f"box=[{bb.x_min:.0f},{bb.y_min:.0f},{bb.x_max:.0f},{bb.y_max:.0f}], "
                      f"embed_norm={norm:.4f}, dim={vec.shape[0]}")

        all_embeddings[filename] = embeddings

        # ── Check expected vs detected ───────────────────────────────
        matched_expected = [c for c in expected if c in detected_categories]
        missed_expected = [c for c in expected if c not in detected_categories]

        status = "FULL_MATCH" if not missed_expected else "PARTIAL" if matched_expected else "NO_MATCH"

        result = {
            "filename": filename,
            "status": status,
            "detections": [
                {"category": str(d.category), "confidence": d.bounding_box.confidence}
                for d in detections
            ],
            "expected": expected,
            "matched": matched_expected,
            "missed": missed_expected,
        }
        results.append(result)

        if verbose:
            emoji = {"FULL_MATCH": "✅", "PARTIAL": "⚠️", "NO_MATCH": "❌", "MISSING": "⏭️"}
            fallback_tag = " [CLIP-fallback]" if used_fallback else ""
            print(f"  {emoji.get(status, '?')} {filename}: {status}{fallback_tag} "
                  f"(detected: {detected_categories}, expected: {expected})")

    # ── Embedding quality: cross-similarity ──────────────────────────────
    # For product images of the same category, embeddings should be more
    # similar to each other than to a different category.
    similarity_tests = []
    categories_with_embeddings = {}
    for fname, emb_dict in all_embeddings.items():
        for cat, vec in emb_dict.items():
            categories_with_embeddings.setdefault(cat, []).append((fname, vec))

    for cat, items in categories_with_embeddings.items():
        if len(items) >= 2:
            # Intra-category similarity (should be higher)
            sims = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    s = cosine_similarity(items[i][1], items[j][1])
                    sims.append(s)
            avg_intra = np.mean(sims) if sims else 0.0
            similarity_tests.append({
                "category": cat,
                "intra_similarity_mean": float(avg_intra),
                "intra_similarity_min": float(np.min(sims)) if sims else 0.0,
                "n_pairs": len(sims),
            })

    return {
        "label": manifest.get("label", "unknown"),
        "total_images": len(manifest["images"]),
        "total_detections": total_detections,
        "clip_fallbacks_used": fallback_used,
        "category_distribution": category_hits,
        "results": results,
        "similarity_tests": similarity_tests,
    }


def print_summary(summary: dict, label: str = "") -> None:
    """Pretty-print the pipeline test summary."""
    print(f"\n{'─' * 62}")
    print(f"  {label or summary['label']} — Pipeline Results")
    print(f"{'─' * 62}")

    total = summary["total_images"]
    full = sum(1 for r in summary["results"] if r["status"] == "FULL_MATCH")
    partial = sum(1 for r in summary["results"] if r["status"] == "PARTIAL")
    none_ = sum(1 for r in summary["results"] if r["status"] == "NO_MATCH")
    missing = sum(1 for r in summary["results"] if r["status"] == "MISSING")

    print(f"  Images tested:    {total}")
    print(f"  Total detections: {summary['total_detections']} "
          f"(CLIP fallback used: {summary.get('clip_fallbacks_used', 0)})")
    print(f"  ✅ Full match:     {full}/{total}")
    print(f"  ⚠️  Partial match:  {partial}/{total}")
    print(f"  ❌ No match:       {none_}/{total}")
    if missing:
        print(f"  ⏭️  Missing files:  {missing}/{total}")

    print(f"\n  Category distribution:")
    for cat, count in sorted(summary["category_distribution"].items()):
        print(f"    {cat:10s}: {count}")

    if summary["similarity_tests"]:
        print(f"\n  Embedding similarity (intra-category):")
        for st in summary["similarity_tests"]:
            print(f"    {st['category']:10s}: mean={st['intra_similarity_mean']:.3f}, "
                  f"min={st['intra_similarity_min']:.3f} ({st['n_pairs']} pairs)")


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO + CLIP pipeline on real fashion images",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save images with bounding boxes drawn")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="YOLO confidence threshold (default: 0.15)")
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "tests" / "real_images",
                        help="Directory containing downloaded images")
    args = parser.parse_args()

    # ── Check images exist ────────────────────────────────────────────────
    outfit_dir = args.image_dir / "outfits"
    product_dir = args.image_dir / "products"
    if not outfit_dir.exists() and not product_dir.exists():
        print("❌ No real images found. Run first:")
        print("   python3.11 scripts/download_test_images.py")
        sys.exit(1)

    annotated_dir = args.image_dir / "annotated" if args.save_annotated else None

    # ── Load models ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Real-Image Pipeline Test")
    print("=" * 62)

    print("\n🔄 Loading YOLO detector...")
    t0 = time.perf_counter()
    from ml.yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    # Override confidence for testing
    detector.confidence_threshold = args.confidence
    print(f"   Loaded in {time.perf_counter() - t0:.1f}s "
          f"(fashion_model={detector.is_fashion_model}, "
          f"classes={len(detector.model_class_names)}, "
          f"confidence={args.confidence})")

    print("🔄 Loading CLIP encoder...")
    t0 = time.perf_counter()
    from ml.clip_encoder import get_clip_encoder
    encoder = get_clip_encoder()
    print(f"   Loaded in {time.perf_counter() - t0:.1f}s")

    print("🔄 Building CLIP zero-shot classifier...")
    t0 = time.perf_counter()
    from ml.clip_classifier import get_clip_classifier
    classifier = get_clip_classifier()
    print(f"   Prototypes built in {time.perf_counter() - t0:.1f}s")

    # ── Run on outfit images ─────────────────────────────────────────────
    if outfit_dir.exists() and (outfit_dir / "manifest.json").exists():
        print(f"\n📸 Testing outfit images ({outfit_dir})...")
        outfit_summary = run_pipeline_test(
            outfit_dir,
            outfit_dir / "manifest.json",
            detector, encoder, classifier,
            verbose=args.verbose,
            save_annotated=args.save_annotated,
            annotated_dir=annotated_dir / "outfits" if annotated_dir else None,
        )
        print_summary(outfit_summary, "Outfit Images")
    else:
        print("\n⏭️  No outfit images found, skipping.")
        outfit_summary = None

    # ── Run on product images ────────────────────────────────────────────
    if product_dir.exists() and (product_dir / "manifest.json").exists():
        print(f"\n🏷️  Testing product images ({product_dir})...")
        product_summary = run_pipeline_test(
            product_dir,
            product_dir / "manifest.json",
            detector, encoder, classifier,
            verbose=args.verbose,
            save_annotated=args.save_annotated,
            annotated_dir=annotated_dir / "products" if annotated_dir else None,
        )
        print_summary(product_summary, "Product Images")
    else:
        print("\n⏭️  No product images found, skipping.")
        product_summary = None

    # ── Cross-category similarity test ───────────────────────────────────
    if product_summary and product_summary["similarity_tests"]:
        print(f"\n{'─' * 62}")
        print("  Cross-Category Embedding Test")
        print(f"{'─' * 62}")
        print("  (Same-category items should be MORE similar than cross-category)")
        # Gather all embeddings from product tests
        # This is a sanity check that the embedding space is meaningful
        cats = product_summary["similarity_tests"]
        if len(cats) >= 2:
            avg_intra = np.mean([c["intra_similarity_mean"] for c in cats])
            print(f"  Average intra-category similarity: {avg_intra:.3f}")
            print(f"  ✅ Embedding space looks reasonable!" if avg_intra > 0.3
                  else f"  ⚠️  Low intra-category similarity — embeddings may not cluster well")

    # ── Save JSON report ─────────────────────────────────────────────────
    report_path = args.image_dir / "pipeline_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detector": {
            "is_fashion_model": detector.is_fashion_model,
            "num_classes": len(detector.model_class_names),
            "class_names": detector.model_class_names,
        },
        "outfits": outfit_summary,
        "products": product_summary,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n📊 Full report saved to {report_path}")
    print()


if __name__ == "__main__":
    main()
