import uuid
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl

# ── Enums ─────────────────────────────────────────────────────────────────────

class GarmentCategory(StrEnum):
    SHIRT = "shirt"
    PANTS = "pants"
    SHOES = "shoes"
    JACKET = "jacket"
    DRESS = "dress"
    SKIRT = "skirt"
    OTHER = "other"


# ── Detection ─────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class DetectedGarment(BaseModel):
    category: GarmentCategory
    bounding_box: BoundingBox
    crop_b64: str | None = None  # base64-encoded cropped image (optional, for debug)


# ── Search ────────────────────────────────────────────────────────────────────

class VectorMatch(BaseModel):
    product_id: str
    score: float
    category: GarmentCategory


# ── Seller ────────────────────────────────────────────────────────────────────

class SellerResponse(BaseModel):
    id: uuid.UUID
    name: str
    city: str | None = None
    country: str | None = None
    website: str | None = None

    model_config = {"from_attributes": True}


# ── Product ───────────────────────────────────────────────────────────────────

class ProductResponse(BaseModel):
    id: uuid.UUID
    name: str
    brand: str | None = None
    category: GarmentCategory
    price: float
    currency: str
    image_url: str | None = None
    product_url: str | None = None
    seller: SellerResponse | None = None
    similarity_score: float = Field(0.0, description="Final ranked score")
    is_local: bool = False

    model_config = {"from_attributes": True}


# ── Detection Response ───────────────────────────────────────────────────────

class ClothingDetectionResponse(BaseModel):
    """Structured detection result with all six clothing categories."""

    shirts: list[DetectedGarment] = Field(
        default_factory=list,
        description="All shirt/top detections above the confidence threshold",
    )
    pants: list[DetectedGarment] = Field(
        default_factory=list,
        description="All pants/trousers detections above the confidence threshold",
    )
    shoes: list[DetectedGarment] = Field(
        default_factory=list,
        description="All footwear detections above the confidence threshold",
    )
    jackets: list[DetectedGarment] = Field(
        default_factory=list,
        description="All jacket/coat detections above the confidence threshold",
    )
    dresses: list[DetectedGarment] = Field(
        default_factory=list,
        description="All dress detections above the confidence threshold",
    )
    skirts: list[DetectedGarment] = Field(
        default_factory=list,
        description="All skirt detections above the confidence threshold",
    )
    total_detections: int = Field(
        0,
        description="Total number of fashion garments found",
    )

    @classmethod
    def from_detections(cls, detections: list["DetectedGarment"]) -> "ClothingDetectionResponse":
        """Build a ClothingDetectionResponse from a flat list of DetectedGarment."""
        shirts  = [d for d in detections if d.category == GarmentCategory.SHIRT]
        pants   = [d for d in detections if d.category == GarmentCategory.PANTS]
        shoes   = [d for d in detections if d.category == GarmentCategory.SHOES]
        jackets = [d for d in detections if d.category == GarmentCategory.JACKET]
        dresses = [d for d in detections if d.category == GarmentCategory.DRESS]
        skirts  = [d for d in detections if d.category == GarmentCategory.SKIRT]
        return cls(
            shirts=shirts,
            pants=pants,
            shoes=shoes,
            jackets=jackets,
            dresses=dresses,
            skirts=skirts,
            total_detections=(
                len(shirts) + len(pants) + len(shoes)
                + len(jackets) + len(dresses) + len(skirts)
            ),
        )


# ── API Request / Response ────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Maximum price per item in USD")
    user_latitude: float | None = Field(None, ge=-90, le=90)
    user_longitude: float | None = Field(None, ge=-180, le=180)
    top_n: int = Field(5, ge=1, le=50, description="Results to return per garment type")


class GarmentRecommendations(BaseModel):
    category: GarmentCategory
    items: list[ProductResponse]


class RecommendationResponse(BaseModel):
    results: list[GarmentRecommendations]
    detected_items: list[GarmentCategory]
    total_matches: int


# ── Catalog Ingestion ─────────────────────────────────────────────────────────

class ProductIngestRequest(BaseModel):
    name: str
    brand: str | None = None
    description: str | None = None
    category: GarmentCategory
    price: float = Field(..., gt=0)
    currency: str = "USD"
    image_url: HttpUrl
    product_url: HttpUrl | None = None
    seller_id: uuid.UUID | None = None


class ProductIngestResponse(BaseModel):
    product_id: uuid.UUID
    vector_id: str
    message: str = "Product indexed successfully"


class ProductUpdateRequest(BaseModel):
    """Fields that may be updated on an existing catalog product."""

    name: str | None = None
    brand: str | None = None
    description: str | None = None
    price: float | None = Field(None, gt=0)
    currency: str | None = None
    image_url: HttpUrl | None = None
    product_url: HttpUrl | None = None
    seller_id: uuid.UUID | None = None


class BulkIngestResult(BaseModel):
    """Per-item outcome from a bulk ingest request."""

    index: int = Field(..., description="0-based position in the original request list")
    success: bool
    product_id: uuid.UUID | None = None
    vector_id: str | None = None
    error: str | None = None


class ProductBulkIngestRequest(BaseModel):
    """Batch product ingestion — up to 100 items per call."""

    products: list[ProductIngestRequest] = Field(..., min_length=1, max_length=100)


class ProductBulkIngestResponse(BaseModel):
    """Aggregated result of a bulk ingest request."""

    results: list[BulkIngestResult]
    total: int
    succeeded: int
    failed: int


# ── Pipeline recommendation ──────────────────────────────────────────────────

class PipelineMatch(BaseModel):
    """A single Pinecone hit returned by the recommendation pipeline."""

    product_id: str = Field(..., description="Vector ID / product UUID in the index")
    score: float = Field(..., description="Cosine similarity score (0–1)")
    metadata: dict = Field(default_factory=dict)

    # Convenience fields surfaced from Pinecone metadata (populated when
    # the catalog was seeded with metadata via IngestionService).
    price:     float | None = Field(None, description="Product price from catalog metadata")
    brand:     str | None   = Field(None, description="Brand name from catalog metadata")
    image_url: str | None   = Field(None, description="Product image URL from catalog metadata")


class PipelineCategoryResult(BaseModel):
    """Top matches for one garment category."""

    category: GarmentCategory
    detection_confidence: float = Field(
        ..., description="YOLO confidence for the highest-confidence crop in this category"
    )
    matches: list[PipelineMatch]


class PipelineRecommendationResponse(BaseModel):
    """Full response from the standalone recommendation pipeline."""

    shirts: list[PipelineCategoryResult] = Field(default_factory=list)
    pants: list[PipelineCategoryResult] = Field(default_factory=list)
    shoes: list[PipelineCategoryResult] = Field(default_factory=list)
    jackets: list[PipelineCategoryResult] = Field(default_factory=list)
    dresses: list[PipelineCategoryResult] = Field(default_factory=list)
    skirts: list[PipelineCategoryResult] = Field(default_factory=list)
    total_detections: int
    total_matches: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str
    pinecone: str
    clip_model_loaded: bool = True
    yolo_model_loaded: bool = True
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
