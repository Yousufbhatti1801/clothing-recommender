"""POST /detect — upload an image and receive per-category garment detections."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.models.schemas import ClothingDetectionResponse
from app.services.detection import DetectionService
from app.utils.image import load_image_from_upload

log = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["Detection"])

UPLOAD_DIR = Path("uploads")


def get_detection_service() -> DetectionService:
    """Dependency: returns a DetectionService backed by the singleton YOLODetector."""
    return DetectionService()


@router.post(
    "",
    response_model=ClothingDetectionResponse,
    summary="Detect clothing items in an uploaded image",
    description=(
        "Accepts a JPEG / PNG / WebP image and returns bounding-box detections "
        "grouped into **shirts**, **pants**, and **shoes** using a fine-tuned "
        "YOLOv8 model."
    ),
)
async def detect_clothing(
    file: UploadFile = File(..., description="Photo of a person wearing clothes"),
    service: DetectionService = Depends(get_detection_service),
) -> ClothingDetectionResponse:
    # ── 1. Validate & pre-process the upload ────────────────────────────────
    image = await load_image_from_upload(file)

    # ── 2. Persist the raw file with a safe, server-generated filename ───────
    #  NEVER use the user-supplied file.filename — it can be a path traversal
    #  attack vector (e.g., "../../etc/cron.d/bad").
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}.jpg"
    dest = UPLOAD_DIR / safe_name
    image.save(dest)
    log.info("Upload saved to %s", dest)

    # ── 3. Run clothing detection ────────────────────────────────────────────
    return await service.detect_clothing(image)
