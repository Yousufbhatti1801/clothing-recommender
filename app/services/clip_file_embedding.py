"""File-based CLIP embedding service.

Loads a singleton CLIP model and produces L2-normalised embeddings for images
provided via filesystem paths. This wraps the existing CLIPEncoder so other
parts of the app can reuse it without reloading the model.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from ml.clip_encoder import CLIPEncoder, get_clip_encoder


class ClipFileEmbeddingService:
    """Generate CLIP embeddings from image file paths."""

    def __init__(self, encoder: CLIPEncoder | None = None) -> None:
        self._encoder = encoder or get_clip_encoder()

    def _load_images(self, paths: Iterable[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in paths:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"Image not found: {p}")
            # Convert to RGB to avoid mode issues (e.g., RGBA, L)
            images.append(Image.open(p).convert("RGB"))
        return images

    def embed_path(self, image_path: str) -> np.ndarray:
        """Embed a single image from disk and return a 512-d float32 vector."""
        embeddings = self.embed_paths([image_path])
        return embeddings[0]

    def embed_paths(self, image_paths: list[str]) -> list[np.ndarray]:
        """Embed multiple images from disk; returns one vector per path."""
        if not image_paths:
            return []
        images = self._load_images(image_paths)
        vectors = self._encoder.encode(images)
        return [vectors[i] for i in range(len(images))]
