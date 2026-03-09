from fastapi import APIRouter
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.core.pinecone_client import get_pinecone_index
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness + readiness check for database and Pinecone connectivity."""
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

    overall = "healthy" if db_status == "ok" and pinecone_status == "ok" else "degraded"
    return HealthResponse(status=overall, database=db_status, pinecone=pinecone_status)
