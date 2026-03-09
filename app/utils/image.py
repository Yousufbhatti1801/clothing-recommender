"""Image loading and preprocessing helpers."""
from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_MB = 10
MAX_DIMENSION = 1024  # pixels — resize longer edge to this before inference


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Validate, read, and pre-process an uploaded image file.

    Raises:
        HTTPException 400 for unsupported formats or oversized files.
    """
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. "
            f"Allowed: {', '.join(SUPPORTED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.",
        )

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    return resize_image(image)


def resize_image(image: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    """Proportionally resize so the longer edge equals max_dim."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)
