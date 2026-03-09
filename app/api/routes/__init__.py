from fastapi import APIRouter
from app.api.routes.image_upload import router as image_upload_router

router = APIRouter()
router.include_router(image_upload_router)