"""POST /pipeline/recommend — full detect → embed → search pipeline in one call."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import PipelineRecommendationResponse
from app.services.recommendation_pipeline import RecommendationPipeline
from app.utils.image import load_image_from_upload

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Module-level singleton — models are loaded once and shared across requests.
_pipeline = RecommendationPipeline()


@router.post(
    "/recommend",
    response_model=PipelineRecommendationResponse,
    summary="Full recommendation pipeline",
    description=(
        "Upload a clothing photo to run the complete AI pipeline:\n\n"
        "1. **YOLOv8 detection** — locates shirts, pants, and shoes.\n"
        "2. **Crop** — extracts each bounding box from the original image.\n"
        "3. **CLIP embedding** — encodes every crop into a 512-d vector.\n"
        "4. **Pinecone search** — finds the top-5 visually similar catalog items per crop.\n"
        "5. **Grouped response** — results are separated by clothing category.\n\n"
        "Supports JPEG, PNG, and WebP; maximum file size 10 MB."
    ),
)
async def recommend(
    file: UploadFile = File(..., description="Photo of a person wearing clothes"),
) -> PipelineRecommendationResponse:
    image = await load_image_from_upload(file)
    return await _pipeline.run(image)
