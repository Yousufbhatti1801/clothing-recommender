"""Unit tests for Pydantic schemas and enums."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    BoundingBox,
    ClothingDetectionResponse,
    DetectedGarment,
    GarmentCategory,
    HealthResponse,
    ProductIngestRequest,
    RecommendationRequest,
    VectorMatch,
)

# ── GarmentCategory ─────────────────────────────────────────────────────────

class TestGarmentCategory:
    def test_all_expected_values_exist(self):
        expected = {"shirt", "pants", "shoes", "jacket", "dress", "skirt", "other"}
        assert {c.value for c in GarmentCategory} == expected

    def test_str_enum_is_string(self):
        assert isinstance(GarmentCategory.SHIRT, str)
        assert GarmentCategory.SHIRT == "shirt"


# ── BoundingBox ──────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_valid_bounding_box(self):
        bb = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200, confidence=0.95)
        assert bb.x_max == 100
        assert bb.confidence == 0.95


# ── DetectedGarment ──────────────────────────────────────────────────────────

class TestDetectedGarment:
    def test_default_crop_b64_is_none(self):
        garment = DetectedGarment(
            category=GarmentCategory.SHIRT,
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1, confidence=0.5),
        )
        assert garment.crop_b64 is None


# ── ClothingDetectionResponse ────────────────────────────────────────────────

class TestClothingDetectionResponse:
    def test_from_detections_groups_correctly(self):
        garments = [
            DetectedGarment(
                category=GarmentCategory.SHIRT,
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.9),
            ),
            DetectedGarment(
                category=GarmentCategory.PANTS,
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.8),
            ),
            DetectedGarment(
                category=GarmentCategory.SHOES,
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.7),
            ),
        ]
        resp = ClothingDetectionResponse.from_detections(garments)
        assert len(resp.shirts) == 1
        assert len(resp.pants) == 1
        assert len(resp.shoes) == 1
        assert resp.total_detections == 3

    def test_from_empty_detections(self):
        resp = ClothingDetectionResponse.from_detections([])
        assert resp.total_detections == 0
        assert resp.shirts == []


# ── RecommendationRequest ────────────────────────────────────────────────────

class TestRecommendationRequest:
    def test_valid_request(self):
        req = RecommendationRequest(budget=100.0, top_n=10)
        assert req.budget == 100.0

    def test_budget_must_be_positive(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(budget=-10.0)

    def test_top_n_bounds(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(budget=100.0, top_n=0)
        with pytest.raises(ValidationError):
            RecommendationRequest(budget=100.0, top_n=51)

    def test_latitude_bounds(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(budget=50.0, user_latitude=91.0)
        with pytest.raises(ValidationError):
            RecommendationRequest(budget=50.0, user_longitude=181.0)


# ── ProductIngestRequest ─────────────────────────────────────────────────────

class TestProductIngestRequest:
    def test_valid_ingest(self):
        req = ProductIngestRequest(
            name="Blue Shirt",
            category=GarmentCategory.SHIRT,
            price=29.99,
            image_url="https://example.com/img.jpg",
        )
        assert req.currency == "USD"  # default

    def test_price_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProductIngestRequest(
                name="Bad Item",
                category=GarmentCategory.SHIRT,
                price=-5,
                image_url="https://example.com/img.jpg",
            )


# ── VectorMatch ──────────────────────────────────────────────────────────────

class TestVectorMatch:
    def test_fields(self):
        m = VectorMatch(product_id="abc-123", score=0.92, category=GarmentCategory.PANTS)
        assert m.score == 0.92


# ── HealthResponse ───────────────────────────────────────────────────────────

class TestHealthResponse:
    def test_defaults(self):
        hr = HealthResponse(status="healthy", database="ok", pinecone="ok")
        assert hr.version == "1.0.0"
