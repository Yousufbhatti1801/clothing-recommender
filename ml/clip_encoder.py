"""CLIP vision encoder — singleton, loaded once at app startup."""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import get_settings
from app.core.executors import get_ml_executor

log = logging.getLogger(__name__)


class CLIPEncoder:
    """Wraps openai/clip-vit-base-patch32 for image embedding."""

    def __init__(self) -> None:
        settings = get_settings()
        self.device = torch.device(settings.clip_device)
        log.info("Loading CLIP model '%s' on device '%s'…", settings.clip_model_name, self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            settings.clip_model_name
        )
        self.model: CLIPModel = CLIPModel.from_pretrained(
            settings.clip_model_name
        ).to(self.device)
        self.model.eval()
        log.info("CLIP model loaded.")

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
        # bfloat16 autocast prevents attention-score overflow → NaN that some
        # fine-tuned CLIP variants (e.g. patrickjohncyh/fashion-clip) exhibit
        # on CPU with float32 arithmetic.  bfloat16 keeps the same exponent
        # range as float32 so the projection weights are numerically stable,
        # then we cast back to float32 for downstream numpy use.
        autocast_ctx = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if self.device.type == "cpu"
            else torch.autocast("cuda", dtype=torch.float16)
        )
        with autocast_ctx:
            features = self.model.get_image_features(**inputs)
        # Back to float32 before L2-normalisation
        features = features.float()
        # Clamp norm to ≥ 1e-8 to prevent division-by-zero → NaN for
        # blank, corrupted, or very small images (e.g. 60×80 Kaggle JPEGs).
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        features = features / norms
        return features.cpu().numpy().astype(np.float32)

    async def encode_async(self, images: list[Image.Image]) -> np.ndarray:
        """Non-blocking wrapper — runs encode() in the dedicated ML thread-pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_ml_executor(), self.encode, images)


@lru_cache(maxsize=1)
def get_clip_encoder() -> CLIPEncoder:
    """Return the process-wide singleton CLIPEncoder (created on first call)."""
    return CLIPEncoder()
