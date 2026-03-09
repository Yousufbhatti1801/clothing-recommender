from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import get_recommendation_service
from app.models.schemas import RecommendationRequest, RecommendationResponse
from app.services.recommendation import RecommendationService
from app.utils.image import load_image_from_upload

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.post("", response_model=RecommendationResponse)
async def recommend(
    file: UploadFile = File(..., description="Image of a person wearing clothes"),
    budget: float = Form(..., gt=0, description="Maximum price per item (USD)"),
    user_latitude: float | None = Form(None),
    user_longitude: float | None = Form(None),
    top_n: int = Form(5, ge=1, le=50),
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """
    Upload a photo and receive visually similar clothing recommendations,
    filtered by budget and ranked by locality.
    """
    image = await load_image_from_upload(file)
    request = RecommendationRequest(
        budget=budget,
        user_latitude=user_latitude,
        user_longitude=user_longitude,
        top_n=top_n,
    )
    return await service.recommend(image, request)
