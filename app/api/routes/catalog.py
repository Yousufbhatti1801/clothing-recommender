from fastapi import APIRouter, Depends

from app.core.dependencies import get_recommendation_service
from app.models.schemas import ProductIngestRequest, ProductIngestResponse
from app.services.ingestion import IngestionService
from app.services.recommendation import RecommendationService
from app.core.pinecone_client import get_pinecone_index
from app.core.dependencies import get_catalog_service, get_embedding_service
from app.services.catalog import CatalogService
from app.services.embedding import EmbeddingService
from pinecone import Index

router = APIRouter(prefix="/catalog", tags=["Catalog"])


def get_ingestion_service(
    catalog: CatalogService = Depends(get_catalog_service),
    embedding: EmbeddingService = Depends(get_embedding_service),
    index: Index = Depends(get_pinecone_index),
) -> IngestionService:
    return IngestionService(catalog=catalog, embedding=embedding, index=index)


@router.post("/ingest", response_model=ProductIngestResponse, status_code=201)
async def ingest_product(
    data: ProductIngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ProductIngestResponse:
    """Embed a new catalog product and store it in Pinecone + PostgreSQL."""
    return await service.ingest(data)
