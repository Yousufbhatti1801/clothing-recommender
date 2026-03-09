from app.api.routes import catalog, detect, health, pipeline, recommendations
from fastapi import APIRouter

router = APIRouter()
router.include_router(recommendations.router)
router.include_router(catalog.router)
router.include_router(detect.router)
router.include_router(pipeline.router)
router.include_router(health.router)
