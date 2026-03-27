"""
ml/clip_classifier.py
======================
CLIP zero-shot garment classifier.

When YOLO fails to detect any garment (e.g. flat-lay product shots, images
where the model has low confidence), this classifier uses CLIP's text-image
similarity to identify what garment category is present in the image.

How it works
------------
1. Pre-compute L2-normalised text embeddings for a set of category prompts
   (e.g. "a photo of a white t-shirt", "a shirt laid flat") once at startup.
2. Per category, average the prompt embeddings into a single prototype vector.
3. For an incoming image embedding (512-d), compute dot-product similarity
   against each category prototype.
4. Return the best-matching GarmentCategory and its confidence score.

Typical use: fallback inside DetectionService when YOLO returns nothing.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from app.core.executors import get_ml_executor
from app.models.schemas import GarmentCategory

log = logging.getLogger(__name__)

# ── Prompt templates per category ────────────────────────────────────────────
# Multiple diverse prompts per category → richer prototype vector.
# Drawn from CLIP's known strengths: product photos, fashion photos, e-commerce.
CATEGORY_PROMPTS: dict[GarmentCategory, list[str]] = {
    GarmentCategory.SHIRT: [
        "a photo of a shirt",
        "a t-shirt",
        "a polo shirt",
        "a dress shirt",
        "a casual top",
        "a blouse",
        "a graphic tee",
        "a hoodie",
        "a sweater",
    ],
    GarmentCategory.PANTS: [
        "a photo of pants",
        "a pair of jeans",
        "trousers",
        "chinos",
        "cargo pants",
        "a photo of denim jeans",
        "slim fit trousers",
    ],
    GarmentCategory.SHOES: [
        "a photo of shoes",
        "a pair of sneakers",
        "leather shoes",
        "running shoes",
        "boots",
        "loafers",
        "a pair of heels",
        "footwear",
    ],
    GarmentCategory.JACKET: [
        "a photo of a jacket",
        "a leather jacket",
        "a denim jacket",
        "a blazer",
        "a coat",
        "a bomber jacket",
        "a windbreaker",
    ],
    GarmentCategory.DRESS: [
        "a photo of a dress",
        "a midi dress",
        "a sundress",
        "a floral dress",
        "a bodycon dress",
        "a maxi dress",
        "a cocktail dress",
    ],
    GarmentCategory.SKIRT: [
        "a photo of a skirt",
        "a mini skirt",
        "a pleated skirt",
        "a midi skirt",
        "a denim skirt",
        "a wrap skirt",
    ],
}


class CLIPClassifier:
    """
    Zero-shot garment classifier using CLIP text-image similarity.

    Loads the CLIP text encoder from the already-instantiated CLIPEncoder
    singleton to avoid loading model weights twice.
    """

    def __init__(self) -> None:
        from ml.clip_encoder import get_clip_encoder
        self._encoder = get_clip_encoder()
        self._prototypes: dict[GarmentCategory, np.ndarray] = {}
        self._build_prototypes()

    # ── Body-region definitions ──────────────────────────────────────────
    # Each region is (y_start_ratio, y_end_ratio, expected_categories).
    # Ratios are proportions of image height, tuned for typical outfit shots
    # where a person fills most of the frame.
    BODY_REGIONS: list[tuple[float, float, list[GarmentCategory]]] = [
        # Upper body — shirt, jacket, dress (top 50%)
        (0.00, 0.50, [GarmentCategory.SHIRT, GarmentCategory.JACKET,
                       GarmentCategory.DRESS]),
        # Lower body — pants, skirt (middle 30-80%)
        (0.35, 0.80, [GarmentCategory.PANTS, GarmentCategory.SKIRT]),
        # Feet — shoes (bottom 25%)
        (0.72, 1.00, [GarmentCategory.SHOES]),
    ]

    def _build_prototypes(self) -> None:
        """Pre-compute and cache category prototype vectors."""
        log.info("Building CLIP zero-shot prototypes for %d categories…",
                 len(CATEGORY_PROMPTS))
        model = self._encoder.model
        processor = self._encoder.processor
        device = self._encoder.device

        with torch.no_grad():
            for category, prompts in CATEGORY_PROMPTS.items():
                inputs = processor(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                text_feats = model.get_text_features(**inputs)
                # L2 normalise each prompt embedding, then average
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                prototype = text_feats.mean(dim=0)
                prototype = prototype / prototype.norm()  # re-normalise the mean
                self._prototypes[category] = prototype.cpu().numpy().astype(np.float32)

        log.info("CLIP zero-shot prototypes ready.")

    def classify(
        self,
        image_embedding: np.ndarray,
        *,
        top_k: int = 1,
    ) -> list[tuple[GarmentCategory, float]]:
        """
        Classify an image by comparing its L2-normalised embedding against
        category prototypes.

        Args:
            image_embedding: 512-d L2-normalised float32 vector.
            top_k: Number of top categories to return (default 1).

        Returns:
            List of (GarmentCategory, cosine_similarity) sorted descending.
        """
        scores: list[tuple[GarmentCategory, float]] = []
        for cat, proto in self._prototypes.items():
            sim = float(np.dot(image_embedding, proto))
            scores.append((cat, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def classify_image(
        self,
        image: Image.Image,
        *,
        top_k: int = 1,
    ) -> list[tuple[GarmentCategory, float]]:
        """
        Convenience method: encode image then classify.

        Args:
            image: PIL Image to classify.
            top_k: Number of top categories to return.

        Returns:
            List of (GarmentCategory, cosine_similarity) sorted descending.
        """
        vec = self._encoder.encode([image])[0]
        return self.classify(vec, top_k=top_k)

    def classify_regions(
        self,
        image: Image.Image,
        *,
        min_confidence: float = 0.20,
    ) -> list[tuple[GarmentCategory, float, Image.Image]]:
        """
        Split an outfit image into body regions and classify each region.

        For each body zone (upper, middle, lower), crops that region, encodes
        it with CLIP, and compares against only the plausible categories for
        that zone. Returns one result per region that passes the confidence
        threshold, along with the cropped image.

        Returns:
            List of (category, confidence, cropped_region) tuples.
            De-duplicated: if two regions both match "jacket", only the
            higher-confidence one is kept.
        """
        w, h = image.size
        seen: dict[GarmentCategory, tuple[float, Image.Image]] = {}

        for y_start_ratio, y_end_ratio, allowed_cats in self.BODY_REGIONS:
            y_start = int(h * y_start_ratio)
            y_end = int(h * y_end_ratio)
            region_crop = image.crop((0, y_start, w, y_end))

            # Encode the cropped region
            region_vec = self._encoder.encode([region_crop])[0]

            # Score against only the allowed categories for this zone
            best_cat: GarmentCategory | None = None
            best_score: float = 0.0
            for cat in allowed_cats:
                proto = self._prototypes.get(cat)
                if proto is None:
                    continue
                sim = float(np.dot(region_vec, proto))
                if sim > best_score:
                    best_score = sim
                    best_cat = cat

            if best_cat is None or best_score < min_confidence:
                continue

            # Keep the highest-confidence detection for each category
            if best_cat in seen:
                if best_score <= seen[best_cat][0]:
                    continue
            seen[best_cat] = (best_score, region_crop)

        return [
            (cat, score, crop) for cat, (score, crop) in seen.items()
        ]

    async def classify_regions_async(
        self,
        image: Image.Image,
        *,
        min_confidence: float = 0.20,
    ) -> list[tuple[GarmentCategory, float, Image.Image]]:
        """Non-blocking wrapper for classify_regions."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_ml_executor(),
            lambda: self.classify_regions(image, min_confidence=min_confidence),
        )

    async def classify_image_async(
        self,
        image: Image.Image,
        *,
        top_k: int = 1,
    ) -> list[tuple[GarmentCategory, float]]:
        """Non-blocking wrapper for classify_image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_ml_executor(),
            lambda: self.classify_image(image, top_k=top_k),
        )


@lru_cache(maxsize=1)
def get_clip_classifier() -> CLIPClassifier:
    """Return the process-wide singleton CLIPClassifier."""
    return CLIPClassifier()
