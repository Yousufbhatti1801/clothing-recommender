from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pinecone import Index

from app.core.auth import require_api_key
from app.core.dependencies import get_catalog_service, get_embedding_service
from app.core.pinecone_client import get_pinecone_index
from app.models.schemas import (
    BulkIngestResult,
    GarmentCategory,
    ProductBulkIngestRequest,
    ProductBulkIngestResponse,
    ProductIngestRequest,
    ProductIngestResponse,
    ProductResponse,
    ProductUpdateRequest,
    SellerResponse,
)
from app.services.catalog import CatalogService
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/catalog", tags=["Catalog"])


def get_ingestion_service(
    catalog: CatalogService = Depends(get_catalog_service),
    embedding: EmbeddingService = Depends(get_embedding_service),
    index: Index = Depends(get_pinecone_index),
) -> IngestionService:
    return IngestionService(catalog=catalog, embedding=embedding, index=index)


# ── Single ingest ────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=ProductIngestResponse,
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
async def ingest_product(
    data: ProductIngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ProductIngestResponse:
    """Embed a new catalog product and store it in Pinecone + PostgreSQL.

    Requires a valid ``X-API-Key`` header.
    """
    return await service.ingest(data)


# ── Bulk ingest ──────────────────────────────────────────────────────────────

@router.post(
    "/ingest/batch",
    response_model=ProductBulkIngestResponse,
    status_code=201,
    dependencies=[Depends(require_api_key)],
    summary="Bulk-ingest up to 100 catalog products",
)
async def ingest_products_batch(
    data: ProductBulkIngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ProductBulkIngestResponse:
    """Ingest multiple products in one request.

    Each item is processed independently — failures in one item do not abort
    the rest.  Inspect the ``results`` array to see per-item outcomes.
    """
    raw = await service.ingest_bulk(data.products)
    results: list[BulkIngestResult] = []
    succeeded = 0
    for i, item in enumerate(raw):
        if isinstance(item, Exception):
            results.append(BulkIngestResult(index=i, success=False, error=str(item)))
        else:
            results.append(
                BulkIngestResult(
                    index=i,
                    success=True,
                    product_id=item.product_id,
                    vector_id=item.vector_id,
                )
            )
            succeeded += 1
    return ProductBulkIngestResponse(
        results=results,
        total=len(raw),
        succeeded=succeeded,
        failed=len(raw) - succeeded,
    )


# ── List products ─────────────────────────────────────────────────────────────

@router.get(
    "/products",
    response_model=list[ProductResponse],
    dependencies=[Depends(require_api_key)],
    summary="List catalog products with optional filtering",
)
async def list_products(
    category: GarmentCategory | None = Query(None, description="Filter by garment category"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    catalog: CatalogService = Depends(get_catalog_service),
) -> list[ProductResponse]:
    """Return a paginated list of catalog products."""
    cat_str = category.value if category else None
    products = await catalog.list_products(category=cat_str, limit=limit, offset=offset)
    return [
        ProductResponse(
            id=p.id,
            name=p.name,
            brand=p.brand,
            category=GarmentCategory(p.category),
            price=p.price,
            currency=p.currency,
            image_url=p.image_url,
            product_url=p.product_url,
            seller=SellerResponse.model_validate(p.seller) if p.seller else None,
            similarity_score=0.0,
        )
        for p in products
    ]


# ── Update product ────────────────────────────────────────────────────────────

@router.patch(
    "/products/{product_id}",
    response_model=ProductResponse,
    dependencies=[Depends(require_api_key)],
    summary="Partially update a product's metadata",
)
async def update_product(
    product_id: uuid.UUID,
    data: ProductUpdateRequest,
    catalog: CatalogService = Depends(get_catalog_service),
) -> ProductResponse:
    """Update one or more metadata fields of an existing product.

    Only the provided fields are changed; omitted fields keep their current values.
    """
    try:
        product = await catalog.update_product(product_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProductResponse(
        id=product.id,
        name=product.name,
        brand=product.brand,
        category=GarmentCategory(product.category),
        price=product.price,
        currency=product.currency,
        image_url=product.image_url,
        product_url=product.product_url,
        seller=SellerResponse.model_validate(product.seller) if product.seller else None,
        similarity_score=0.0,
    )


# ── Delete product ────────────────────────────────────────────────────────────

@router.delete(
    "/products/{product_id}",
    status_code=204,
    response_model=None,
    dependencies=[Depends(require_api_key)],
    summary="Delete a product from PostgreSQL and Pinecone",
)
async def delete_product(
    product_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> None:
    """Permanently remove a product and its Pinecone vector."""
    try:
        await service.delete_product(product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
