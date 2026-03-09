"""POST /detect — upload an image and receive per-category garment detections."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.models.schemas import ClothingDetectionResponse
from app.services.detection import DetectionService
from app.utils.image import load_image_from_upload

router = APIRouter(prefix="/detect", tags=["Detection"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
        "YOLOv8 model.  The uploaded file is persisted under `uploads/`."
    ),
)
async def detect_clothing(
    file: UploadFile = File(..., description="Photo of a person wearing clothes"),
    service: DetectionService = Depends(get_detection_service),
) -> ClothingDetectionResponse:
    # ── 1. Validate & pre-process the upload ────────────────────────────────
    image = await load_image_from_upload(file)

    # ── 2. Persist the raw file ──────────────────────────────────────────────
    #  Re-read content from the PIL image so we don't need a second file.read()
    dest = UPLOAD_DIR / (file.filename or "upload.jpg")
    image.save(dest)

    # ── 3. Run clothing detection ────────────────────────────────────────────
    return await service.detect_clothing(image)
