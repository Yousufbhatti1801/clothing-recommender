"""POST /upload/image — validate and persist an uploaded image with a secure UUID filename."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.utils.image import load_image_from_upload

log = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Image Upload"])

UPLOAD_DIR = Path("uploads")


@router.post(
    "/image",
    summary="Upload and persist an image",
    description=(
        "Validates the uploaded file (JPEG / PNG / WebP, max 10 MB), resizes it to "
        "at most 1024 px on the longer edge, and saves it under a server-generated "
        "UUID filename.  The original client-supplied filename is **never used** to "
        "prevent path-traversal attacks."
    ),
)
async def upload_image(
    file: UploadFile = File(..., description="JPEG / PNG / WebP image to upload"),
) -> dict:
    """Upload and persist an image; returns the secure server-side filename."""
    image = await load_image_from_upload(file)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}.jpg"
    dest = UPLOAD_DIR / safe_name
    image.save(dest, format="JPEG", quality=95)
    log.info("Upload saved to %s", dest)
    return {"message": "File uploaded successfully", "filename": safe_name}