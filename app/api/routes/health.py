import time

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.core.pinecone_client import get_pinecone_index
from app.models.schemas import HealthResponse
from ml.clip_encoder import get_clip_encoder
from ml.yolo_detector import get_yolo_detector

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Liveness + readiness check for database, Pinecone, and ML models.

    Returns HTTP 503 when any critical dependency is unavailable,
    so that Kubernetes readiness probes and AWS Target Group health
    checks correctly mark the instance as unhealthy.
    """
    db_status = "ok"
    pinecone_status = "ok"

    # Check PostgreSQL
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    # Check Pinecone
    try:
        index = get_pinecone_index()
        index.describe_index_stats()
    except Exception as exc:
        pinecone_status = f"error: {exc}"

    # Check ML models (singletons should already be loaded)
    clip_loaded = True
    yolo_loaded = True
    try:
        get_clip_encoder()
    except Exception:
        clip_loaded = False
    try:
        get_yolo_detector()
    except Exception:
        yolo_loaded = False

    overall = (
        "healthy"
        if db_status == "ok" and pinecone_status == "ok" and clip_loaded and yolo_loaded
        else "degraded"
    )

    # Return 503 so orchestrators (K8s, ECS) can route traffic away
    if overall != "healthy":
        response.status_code = 503

    start_time = getattr(request.app.state, "start_time", time.monotonic())
    uptime = time.monotonic() - start_time

    return HealthResponse(
        status=overall,
        database=db_status,
        pinecone=pinecone_status,
        clip_model_loaded=clip_loaded,
        yolo_model_loaded=yolo_loaded,
        uptime_seconds=round(uptime, 1),
    )
