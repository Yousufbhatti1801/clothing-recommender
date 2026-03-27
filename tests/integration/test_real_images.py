"""
tests/integration/test_real_images.py
======================================
Integration tests that exercise the YOLO + CLIP pipeline on *real* fashion
photographs downloaded from Unsplash.

These tests are **skipped automatically** when images aren't present.
To enable them:

    python3.11 scripts/download_test_images.py
    PINECONE_API_KEY=dummy python3.11 -m pytest tests/integration/test_real_images.py -v

What these tests verify
-----------------------
1. YOLO produces *some* detections on real outfit photos (even at low conf).
2. CLIP produces valid 512-d L2-normalised embeddings from real images.
3. Same-category product embeddings are more similar to each other than
   to a random noise vector (embedding space sanity check).
4. Detection bounding boxes are within image bounds.
5. No crashes on diverse real-world photos (varying aspect ratios,
   lighting, backgrounds).

Model Note
----------
The current YOLO model (keremberke/yolov8n-fashion-detection, nano) has low
recall on general Unsplash photos.  Tests use a relaxed confidence threshold
(0.15) and lenient pass criteria.  After fine-tuning with
``scripts/train_fashion_yolo.py``, these tests should pass with stricter
thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

OUTFIT_DIR = ROOT / "tests" / "real_images" / "outfits"
PRODUCT_DIR = ROOT / "tests" / "real_images" / "products"

# ── Skip conditions ──────────────────────────────────────────────────────────

has_outfits = OUTFIT_DIR.exists() and (OUTFIT_DIR / "manifest.json").exists()
has_products = PRODUCT_DIR.exists() and (PRODUCT_DIR / "manifest.json").exists()

skip_no_outfits = pytest.mark.skipif(
    not has_outfits,
    reason="No real outfit images. Run: python3.11 scripts/download_test_images.py",
)
skip_no_products = pytest.mark.skipif(
    not has_products,
    reason="No real product images. Run: python3.11 scripts/download_test_images.py",
)
skip_no_images = pytest.mark.skipif(
    not (has_outfits or has_products),
    reason="No real images. Run: python3.11 scripts/download_test_images.py",
)

# Relaxed confidence for the nano model — raise after fine-tuning
YOLO_TEST_CONFIDENCE = 0.15

# ── Fixtures (real ML models — loaded once per module) ───────────────────────

@pytest.fixture(scope="module")
def yolo_detector():
    """Load the real YOLO detector (downloads weights if needed)."""
    import os
    os.environ.setdefault("PINECONE_API_KEY", "dummy")
    from ml.yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    # Override confidence for testing
    detector.confidence_threshold = YOLO_TEST_CONFIDENCE
    return detector


@pytest.fixture(scope="module")
def clip_encoder():
    """Load the real CLIP encoder (downloads from HuggingFace if needed)."""
    import os
    os.environ.setdefault("PINECONE_API_KEY", "dummy")
    from ml.clip_encoder import get_clip_encoder
    return get_clip_encoder()


@pytest.fixture(scope="module")
def outfit_manifest() -> list[dict]:
    if not has_outfits:
        return []
    with open(OUTFIT_DIR / "manifest.json") as f:
        return json.load(f)["images"]


@pytest.fixture(scope="module")
def product_manifest() -> list[dict]:
    if not has_products:
        return []
    with open(PRODUCT_DIR / "manifest.json") as f:
        return json.load(f)["images"]


def _load_images(directory: Path, manifest: list[dict]) -> list[tuple[str, Image.Image]]:
    """Load all images from manifest that exist on disk."""
    images = []
    for entry in manifest:
        path = directory / entry["filename"]
        if path.exists():
            images.append((entry["filename"], Image.open(path).convert("RGB")))
    return images


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO Detection Tests (outfit images)
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_outfits
class TestYOLOOnRealOutfits:
    """Test YOLO detection on real full-body outfit photos."""

    def test_detects_garments_in_some_images(
        self, yolo_detector, outfit_manifest,
    ):
        """
        At least SOME outfit images should produce garment detections.

        The nano model won't detect garments in every photo, but if it
        fails on ALL of them, something is fundamentally broken.
        """
        detected_count = 0
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            detections = yolo_detector.detect_all_fashion(img)
            if len(detections) > 0:
                detected_count += 1

        # At least 30% of images should have at least one detection
        total = sum(1 for e in outfit_manifest if (OUTFIT_DIR / e["filename"]).exists())
        assert total > 0, "No outfit images found on disk"
        min_expected = max(1, int(total * 0.3))
        assert detected_count >= min_expected, (
            f"YOLO detected garments in only {detected_count}/{total} images "
            f"(need at least {min_expected})"
        )

    def test_detections_have_valid_bounding_boxes(
        self, yolo_detector, outfit_manifest,
    ):
        """All bounding boxes should be within image bounds."""
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            detections = yolo_detector.detect_all_fashion(img)
            for det in detections:
                bb = det.bounding_box
                assert 0 <= bb.x_min <= w, f"{entry['filename']}: x_min={bb.x_min}"
                assert 0 <= bb.y_min <= h, f"{entry['filename']}: y_min={bb.y_min}"
                assert bb.x_min < bb.x_max, f"{entry['filename']}: degenerate width"
                assert bb.y_min < bb.y_max, f"{entry['filename']}: degenerate height"
                assert 0 < bb.confidence <= 1.0, f"confidence={bb.confidence}"

    def test_model_does_not_crash_on_any_image(
        self, yolo_detector, outfit_manifest,
    ):
        """No image should cause a crash — even if it produces zero detections."""
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            # Should not raise
            detections = yolo_detector.detect_all_fashion(img)
            assert isinstance(detections, list)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP Embedding Tests (product images — direct, no YOLO dependency)
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_products
class TestCLIPOnRealProducts:
    """Test CLIP embedding on real product photographs."""

    def test_embedding_shape_and_normalisation(
        self, clip_encoder, product_manifest,
    ):
        """Each product image should produce a 512-d L2-normalised vector."""
        for entry in product_manifest:
            img_path = PRODUCT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            vec = clip_encoder.encode([img])[0]

            assert vec.shape == (512,), f"{entry['filename']}: shape={vec.shape}"
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 0.05, (
                f"{entry['filename']}: L2 norm = {norm:.4f} (expected ~1.0)"
            )

    def test_batch_encoding_works(
        self, clip_encoder, product_manifest,
    ):
        """Batch encoding multiple images at once produces correct shapes."""
        images = []
        for entry in product_manifest[:4]:
            img_path = PRODUCT_DIR / entry["filename"]
            if img_path.exists():
                images.append(Image.open(img_path).convert("RGB"))

        assert len(images) >= 2, "Need at least 2 images for batch test"
        batch_vecs = clip_encoder.encode(images)
        assert batch_vecs.shape == (len(images), 512)
        norms = np.linalg.norm(batch_vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=0.05)

    def test_different_images_produce_different_embeddings(
        self, clip_encoder, product_manifest,
    ):
        """No two product images should have identical embeddings."""
        embeddings = []
        filenames = []
        for entry in product_manifest:
            img_path = PRODUCT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            vec = clip_encoder.encode([img])[0]
            embeddings.append(vec)
            filenames.append(entry["filename"])

        assert len(embeddings) >= 2, "Need at least 2 product images"
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                assert sim < 0.999, (
                    f"{filenames[i]} and {filenames[j]} are nearly identical "
                    f"(cosine sim = {sim:.4f})"
                )

    def test_same_category_more_similar_than_noise(
        self, clip_encoder, product_manifest,
    ):
        """
        Same-category embeddings should be more similar to each other
        than to a random noise vector.
        """
        rng = np.random.RandomState(42)
        noise = rng.randn(512).astype(np.float32)
        noise /= np.linalg.norm(noise)

        by_category: dict[str, list[np.ndarray]] = {}
        for entry in product_manifest:
            img_path = PRODUCT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            vec = clip_encoder.encode([img])[0]
            cat = entry["expected_garments"][0]
            by_category.setdefault(cat, []).append(vec)

        for cat, vecs in by_category.items():
            if len(vecs) < 2:
                continue
            sims = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append(float(np.dot(vecs[i], vecs[j])))
            avg_intra = np.mean(sims)

            noise_sims = [float(np.dot(v, noise)) for v in vecs]
            avg_noise = np.mean(noise_sims)

            assert avg_intra > avg_noise, (
                f"Category '{cat}': intra-sim ({avg_intra:.3f}) should be > "
                f"noise-sim ({avg_noise:.3f})"
            )

    def test_shoes_cluster_away_from_shirts(
        self, clip_encoder, product_manifest,
    ):
        """
        Shoe embeddings should be less similar to shirt embeddings than
        to other shoe embeddings — validates CLIP semantic awareness.
        """
        shoes_vecs = []
        shirt_vecs = []
        for entry in product_manifest:
            img_path = PRODUCT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            vec = clip_encoder.encode([img])[0]
            cat = entry["expected_garments"][0]
            if cat == "shoes":
                shoes_vecs.append(vec)
            elif cat == "shirt":
                shirt_vecs.append(vec)

        if len(shoes_vecs) < 2 or len(shirt_vecs) < 1:
            pytest.skip("Not enough shoe/shirt images for cross-category test")

        # Intra-shoe similarity
        shoe_sims = []
        for i in range(len(shoes_vecs)):
            for j in range(i + 1, len(shoes_vecs)):
                shoe_sims.append(float(np.dot(shoes_vecs[i], shoes_vecs[j])))
        avg_shoe_intra = np.mean(shoe_sims)

        # Cross shoe-shirt similarity
        cross_sims = []
        for sv in shoes_vecs:
            for shv in shirt_vecs:
                cross_sims.append(float(np.dot(sv, shv)))
        avg_cross = np.mean(cross_sims)

        # Shoes should be more similar to each other than to shirts
        assert avg_shoe_intra > avg_cross, (
            f"Shoe-shoe sim ({avg_shoe_intra:.3f}) should be > "
            f"shoe-shirt sim ({avg_cross:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Outfit CLIP Tests (embed whole outfit images directly)
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_outfits
class TestCLIPOnOutfits:
    """Test CLIP embedding on full outfit photos (bypassing YOLO)."""

    def test_outfit_embeddings_are_valid(
        self, clip_encoder, outfit_manifest,
    ):
        """Full outfit photos produce valid 512-d embeddings."""
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            vec = clip_encoder.encode([img])[0]
            assert vec.shape == (512,)
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 0.05

    def test_outfit_embeddings_are_distinct(
        self, clip_encoder, outfit_manifest,
    ):
        """Each outfit should produce a unique embedding."""
        embeddings = []
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            embeddings.append(clip_encoder.encode([img])[0])

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                assert sim < 0.99, f"Outfits {i} and {j} too similar: {sim:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Pipeline Tests (YOLO → crop → CLIP)
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_outfits
class TestFullPipeline:
    """End-to-end: detect garments in outfit photos, then embed the crops."""

    def test_detect_and_embed_pipeline(
        self, yolo_detector, clip_encoder, outfit_manifest,
    ):
        """
        For images where YOLO finds garments:
        1. Crop each detection
        2. CLIP embeds each crop → valid 512-d vector

        At least one image should pass through the full pipeline.
        """
        processed = 0
        for entry in outfit_manifest:
            img_path = OUTFIT_DIR / entry["filename"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            detections = yolo_detector.detect_all_fashion(img)
            if not detections:
                continue

            crops = []
            for det in detections:
                bb = det.bounding_box
                crop = img.crop((int(bb.x_min), int(bb.y_min),
                                 int(bb.x_max), int(bb.y_max)))
                if crop.size[0] >= 10 and crop.size[1] >= 10:
                    crops.append(crop)

            if not crops:
                continue

            embeddings = clip_encoder.encode(crops)
            assert embeddings.shape == (len(crops), 512)
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=0.05)
            processed += 1

        assert processed >= 1, (
            "Expected at least 1 outfit image to pass full YOLO→CLIP pipeline"
        )
