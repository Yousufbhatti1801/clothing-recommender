from typing import Annotated

from fastapi import Depends
from pinecone import Index
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.pinecone_client import get_pinecone_index
from app.services.catalog import CatalogService
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.recommendation import RecommendationService
from app.services.search import SearchService

# ── Database ──────────────────────────────────────────────────────────────────

DbSession = Annotated[AsyncSession, Depends(get_db)]
PineconeIndex = Annotated[Index, Depends(get_pinecone_index)]

# ── Services ──────────────────────────────────────────────────────────────────


def get_detection_service() -> DetectionService:
    return DetectionService()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_search_service(index: PineconeIndex) -> SearchService:
    return SearchService(index=index)


def get_catalog_service(db: DbSession) -> CatalogService:
    return CatalogService(db=db)


def get_recommendation_service(
    db: DbSession,
    index: PineconeIndex,
) -> RecommendationService:
    return RecommendationService(
        detection=DetectionService(),
        embedding=EmbeddingService(),
        search=SearchService(index=index),
        catalog=CatalogService(db=db),
    )
