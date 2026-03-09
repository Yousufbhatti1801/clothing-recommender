"""CLIP vision encoder — singleton, loaded once at app startup."""
from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import get_settings


class CLIPEncoder:
    """Wraps openai/clip-vit-base-patch32 for image embedding."""

    def __init__(self) -> None:
        settings = get_settings()
        self.device = torch.device(settings.clip_device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            settings.clip_model_name
        )
        self.model: CLIPModel = CLIPModel.from_pretrained(
            settings.clip_model_name
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """
        Encode a batch of PIL Images into L2-normalised 512-d vectors.

        Args:
            images: List of PIL Image objects (RGB).

        Returns:
            np.ndarray of shape (N, 512), dtype float32.
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            self.device
        )
        features = self.model.get_image_features(**inputs)
        # L2 normalise so cosine similarity == dot product
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    async def encode_async(self, images: list[Image.Image]) -> np.ndarray:
        """Non-blocking wrapper — runs encode() in the default thread-pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, images)


@lru_cache(maxsize=1)
def get_clip_encoder() -> CLIPEncoder:
    """Return the process-wide singleton CLIPEncoder (created on first call)."""
    return CLIPEncoder()
