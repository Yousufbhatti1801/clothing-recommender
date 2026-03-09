"""EmbeddingService: encodes PIL Image crops → float32 vectors via CLIP."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ml.clip_encoder import CLIPEncoder, get_clip_encoder


class EmbeddingService:
    def __init__(self, encoder: CLIPEncoder | None = None) -> None:
        self._encoder = encoder or get_clip_encoder()

    async def embed(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """
        Embed a list of image crops.

        Returns:
            List of 512-d float32 numpy arrays (one per crop), L2-normalised.
        """
        if not crops:
            return []
        embeddings = await self._encoder.encode_async(crops)
        return [embeddings[i] for i in range(len(crops))]

    async def embed_single(self, crop: Image.Image) -> np.ndarray:
        results = await self.embed([crop])
        return results[0]
