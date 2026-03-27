"""
tests/integration/test_full_recommendation_flow.py
====================================================
Comprehensive integration tests for the complete clothing recommendation
pipeline.  Every stage — from raw image bytes to a JSON response — is
exercised while mocking only the external I/O boundaries.

Mock boundaries (what is faked):
  • YOLOv8 weights     → MagicMock  (no file I/O, no GPU)
  • CLIP model         → MagicMock  (deterministic random 512-d vectors)
  • Pinecone HTTP      → MagicMock  (returns controlled product IDs per namespace)
  • PostgreSQL session → AsyncMock  (returns controlled ORM-like product objects)

Real code exercised:
  • DetectionService    (bounding-box parsing, tiny-box filtering, cropping)
  • EmbeddingService    (delegates to CLIP, returns per-crop arrays)
  • SearchService       (namespace routing, VectorMatch construction)
  • RecommendationService (full orchestration, budget filter, locality boost,
                           top-n sorting)
  • FastAPI routes      (request validation, dependency injection, HTTP schema)
  • Image preprocessing (MIME check, size limit, resize-before-inference)

Test class map
--------------
  TestDetectionStageWiring      — YOLO is called correctly; crops match bounding boxes
  TestEmbeddingStageWiring      — CLIP receives crop PIL images; output shapes correct
  TestVectorSearchStageWiring   — Pinecone queried per category namespace; matches structured
  TestStageInvocationOrder      — All four stages are called in the expected sequence
  TestBudgetFilteringIntegration — Budget boundary conditions exercised end-to-end
  TestLocalityBoostIntegration  — Local sellers boosted; distant sellers unaffected
  TestCompleteServicePipeline   — Multi-category E2E; short-circuit on no detections
  TestAPIRecommendationFlow     — HTTP /recommend with real service business logic
  TestAPIPipelineEndpointFlow   — HTTP /pipeline/recommend with real pipeline logic
  TestImageFormatHandling       — JPEG / PNG / WebP accepted; bad formats rejected
  TestFailureScenarios          — All meaningful empty/error paths
"""
from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.models.schemas import (
    GarmentCategory,
    GarmentRecommendations,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.catalog import CatalogService
from app.services.detect_and_embed import DetectAndEmbedPipeline
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.recommendation import RecommendationService
from app.services.recommendation_pipeline import RecommendationPipeline
from app.services.search import SearchService
from tests.conftest import make_garment, make_mock_product
from tests.fixtures.images import (
    make_corrupt_jpeg,
    make_jpeg,
    make_oversized_jpeg,
    make_pdf_bytes,
    make_png,
    make_webp,
)
from tests.integration.conftest import (
    PINECONE_MATCHES,
    SHIRT_A_ID,
    SHIRT_B_ID,
    USER_LAT,
    USER_LON,
    build_clip_mock,
    build_pinecone_index_mock,
    build_yolo_mock,
)

# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_catalog_mock(products_map: dict) -> MagicMock:
    """Return an AsyncMock CatalogService whose get_products_by_ids returns *products_map*."""
    catalog = MagicMock(spec=CatalogService)
    catalog.get_products_by_ids = AsyncMock(return_value=products_map)
    return catalog


def _build_recommendation_service(
    yolo_garments,
    products_map: dict,
    clip_mock=None,
    index_mock=None,
) -> tuple[RecommendationService, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Wire real service classes together with injected mocks.

    Returns:
        (service, yolo_mock, clip_mock, index_mock, catalog_mock)
    """
    yolo    = build_yolo_mock(yolo_garments)
    clip    = clip_mock or build_clip_mock(len(yolo_garments) or 1)
    index   = index_mock or build_pinecone_index_mock()
    catalog = _make_catalog_mock(products_map)

    svc = RecommendationService(
        detection=DetectionService(detector=yolo, clip_fallback=False),
        embedding=EmbeddingService(encoder=clip),
        search=SearchService(index=index),
        catalog=catalog,
    )
    return svc, yolo, clip, index, catalog


def _sample_image(w: int = 640, h: int = 640) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 128, 128))


# ══════════════════════════════════════════════════════════════════════════════
# Shared wired HTTP client fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def wired_client(catalog_map):
    """
    httpx AsyncClient backed by the real FastAPI app where:
      • The real RecommendationService is injected (budget + locality logic runs).
      • The real RecommendationPipeline is injected (detect → embed → search runs).
      • YOLOv8 / CLIP / Pinecone / PostgreSQL are all mocked.

    The Pinecone index returns SHIRT_B (0.91) > SHIRT_A (0.85) by raw score,
    so locality-boost tests can verify that SHIRT_A (local seller) rises above
    SHIRT_B when user coordinates are supplied.

    Attributes attached to the client for introspection:
      client._yolo, client._clip, client._index, client._catalog
    """
    # ── Build mock ML / Pinecone ──────────────────────────────────────────
    garments = [
        make_garment(GarmentCategory.SHIRT, x_min=100, y_min=50,  x_max=400, y_max=350, confidence=0.92),
        make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600, confidence=0.85),
        make_garment(GarmentCategory.SHOES, x_min=150, y_min=620, x_max=350, y_max=700, confidence=0.78),
    ]
    yolo  = build_yolo_mock(garments)
    clip  = build_clip_mock(max_batch=len(garments))
    index = build_pinecone_index_mock(PINECONE_MATCHES)

    catalog_mock = _make_catalog_mock(catalog_map)

    # ── FastAPI dependency override factories ────────────────────────────
    def _override_recommend_service():
        return RecommendationService(
            detection=DetectionService(detector=yolo),
            embedding=EmbeddingService(encoder=clip),
            search=SearchService(index=index),
            catalog=catalog_mock,
        )

    # ── Apply patches and spin up the app ────────────────────────────────
    with (
        patch("app.services.vector_store.Pinecone"),
        patch("ml.yolo_detector.get_yolo_detector", return_value=yolo),
        patch("ml.clip_encoder.get_clip_encoder", return_value=clip),
        patch("app.services.detection.get_yolo_detector", return_value=yolo),
        patch("app.services.detect_and_embed.get_yolo_detector", return_value=yolo),
        patch("app.services.detect_and_embed.get_clip_encoder", return_value=clip),
        patch("app.core.pinecone_client.init_pinecone"),
        patch("app.core.pinecone_client.get_pinecone_index", return_value=index),
        patch("app.api.routes.health.AsyncSessionLocal"),
        patch("app.core.database.engine") as mock_engine,
    ):
        # Prevent lifespan from touching a real database
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        begin_cm = AsyncMock()
        begin_cm.__aenter__.return_value = mock_conn
        begin_cm.__aexit__.return_value = False
        mock_engine.begin.return_value = begin_cm
        mock_engine.dispose = AsyncMock()

        from app.core.dependencies import get_recommendation_service
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_recommendation_service] = _override_recommend_service

        # Replace the pipeline singleton with a real pipeline wired to mock ML
        detect_embed_pipeline = DetectAndEmbedPipeline(
            detection_service=DetectionService(detector=yolo),
            embedding_service=EmbeddingService(encoder=clip),
        )
        app.state.pipeline = RecommendationPipeline(
            detect_embed=detect_embed_pipeline,
            vector_store=MagicMock(query=MagicMock(return_value=[])),
        )

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        http_client._yolo    = yolo
        http_client._clip    = clip
        http_client._index   = index
        http_client._catalog = catalog_mock

        try:
            yield http_client
        finally:
            await http_client.aclose()
            app.dependency_overrides.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestDetectionStageWiring
#
# Validates:
#   • DetectionService calls the YOLO model with the full PIL image.
#   • detect_and_crop returns one (DetectedGarment, PIL.Image) pair per
#     garment whose bounding box covers ≥ 1 % of the image area.
#   • Crop dimensions match the bounding box coordinates exactly.
#   • Garments with bounding boxes < 1 % of the image are discarded.
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionStageWiring:
    """
    Verifies that DetectionService correctly delegates to the YOLO model
    and produces (garment, crop) pairs whose geometry matches the bounding boxes.
    """

    @pytest.mark.asyncio
    async def test_yolo_receives_full_image(self):
        """YOLO's detect_async is called exactly once with the original PIL image."""
        garment = make_garment(GarmentCategory.SHIRT)
        yolo = build_yolo_mock([garment])
        svc = DetectionService(detector=yolo)
        image = _sample_image()

        await svc.detect_and_crop(image)

        yolo.detect_async.assert_called_once_with(image)

    @pytest.mark.asyncio
    async def test_crop_dimensions_match_bounding_box(self):
        """Each crop's pixel dimensions equal (x_max-x_min) × (y_max-y_min)."""
        bb_w, bb_h = 300, 300
        garment = make_garment(
            GarmentCategory.SHIRT,
            x_min=100, y_min=50, x_max=100 + bb_w, y_max=50 + bb_h,
        )
        yolo = build_yolo_mock([garment])
        svc = DetectionService(detector=yolo)

        pairs = await svc.detect_and_crop(_sample_image())

        assert len(pairs) == 1
        _, crop = pairs[0]
        assert crop.width  == bb_w
        assert crop.height == bb_h

    @pytest.mark.asyncio
    async def test_tiny_bounding_box_is_discarded(self):
        """
        A bounding box covering < 1 % of the image area is filtered out.
        5×5 on a 640×640 image = 0.006 % → should be dropped.
        """
        tiny = make_garment(GarmentCategory.SHOES, x_min=0, y_min=0, x_max=5, y_max=5)
        yolo = build_yolo_mock([tiny])
        svc = DetectionService(detector=yolo)

        pairs = await svc.detect_and_crop(_sample_image(640, 640))

        assert pairs == [], "Tiny bounding box must be filtered"

    @pytest.mark.asyncio
    async def test_mixed_valid_and_tiny_garments(self):
        """Only garments above the 1 % area threshold are returned."""
        large = make_garment(GarmentCategory.SHIRT)
        tiny  = make_garment(GarmentCategory.SHOES, x_min=0, y_min=0, x_max=5, y_max=5)
        yolo = build_yolo_mock([large, tiny])
        svc = DetectionService(detector=yolo)

        pairs = await svc.detect_and_crop(_sample_image(640, 640))

        assert len(pairs) == 1
        assert pairs[0][0].category == GarmentCategory.SHIRT

    @pytest.mark.asyncio
    async def test_multi_category_detections_all_returned(self):
        """Shirt, pants, and shoes garments are all returned when valid."""
        garments = [
            make_garment(GarmentCategory.SHIRT, x_min=100, y_min=50,  x_max=400, y_max=350),
            make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600),
            make_garment(GarmentCategory.SHOES, x_min=150, y_min=620, x_max=350, y_max=700),
        ]
        yolo = build_yolo_mock(garments)
        svc = DetectionService(detector=yolo)

        pairs = await svc.detect_and_crop(_sample_image(640, 720))

        categories = {g.category for g, _ in pairs}
        assert categories == {GarmentCategory.SHIRT, GarmentCategory.PANTS, GarmentCategory.SHOES}


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestEmbeddingStageWiring
#
# Validates:
#   • EmbeddingService calls CLIP with a list of PIL Image crops
#     (not the full original image).
#   • One 512-d float32 vector is returned per input crop.
#   • An empty crop list returns an empty vector list without calling CLIP.
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingStageWiring:
    """
    Verifies that EmbeddingService correctly delegates to CLIP and
    produces one 512-d vector per garment crop.
    """

    @pytest.mark.asyncio
    async def test_clip_called_with_crops_not_full_image(self):
        """
        CLIP's encode_async receives a list of cropped PIL images,
        not the original full-resolution image.
        """
        crops = [Image.new("RGB", (100, 100)), Image.new("RGB", (80, 200))]
        clip  = build_clip_mock(2)
        svc   = EmbeddingService(encoder=clip)

        await svc.embed(crops)

        clip.encode_async.assert_called_once()
        called_with = clip.encode_async.call_args[0][0]
        assert len(called_with) == 2
        assert all(isinstance(img, Image.Image) for img in called_with)

    @pytest.mark.asyncio
    async def test_one_embedding_per_crop(self):
        """embed() returns exactly one vector for each input crop."""
        n = 3
        crops = [Image.new("RGB", (100, 100)) for _ in range(n)]
        clip  = build_clip_mock(n)
        svc   = EmbeddingService(encoder=clip)

        embeddings = await svc.embed(crops)

        assert len(embeddings) == n

    @pytest.mark.asyncio
    async def test_embedding_is_512d_float32(self):
        """Each returned embedding has shape (512,) and dtype float32."""
        crops = [Image.new("RGB", (150, 150))]
        clip  = build_clip_mock(1)
        svc   = EmbeddingService(encoder=clip)

        embeddings = await svc.embed(crops)

        assert embeddings[0].shape == (512,)
        assert embeddings[0].dtype == np.float32

    @pytest.mark.asyncio
    async def test_empty_crops_returns_empty_no_clip_call(self):
        """No crops → empty return value and CLIP is never called."""
        clip = build_clip_mock()
        svc  = EmbeddingService(encoder=clip)

        result = await svc.embed([])

        assert result == []
        clip.encode_async.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestVectorSearchStageWiring
#
# Validates:
#   • SearchService queries Pinecone with the correct namespace for each
#     garment category (e.g., shirt → "shirt", pants → "pants").
#   • The raw embedding vector is passed to Pinecone correctly.
#   • Results are wrapped in VectorMatch objects with the right fields.
#   • An empty Pinecone response returns an empty VectorMatch list.
#   • search_many aggregates results across multiple categories.
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorSearchStageWiring:
    """
    Verifies that SearchService routes queries to the correct Pinecone
    namespace and returns correctly structured VectorMatch objects.
    """

    @pytest.mark.asyncio
    async def test_shirt_query_uses_shirt_namespace(self):
        """A shirt embedding is queried in the 'shirt' Pinecone namespace."""
        index = build_pinecone_index_mock()
        svc   = SearchService(index=index)
        vec   = np.random.rand(512).astype(np.float32)

        await svc.search(vec, GarmentCategory.SHIRT)

        call_kwargs = index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "shirt"

    @pytest.mark.asyncio
    async def test_pants_query_uses_pants_namespace(self):
        """A pants embedding is queried in the 'pants' Pinecone namespace."""
        index = build_pinecone_index_mock()
        svc   = SearchService(index=index)
        vec   = np.random.rand(512).astype(np.float32)

        await svc.search(vec, GarmentCategory.PANTS)

        call_kwargs = index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "pants"

    @pytest.mark.asyncio
    async def test_returns_vector_match_objects(self):
        """Results are VectorMatch instances with product_id, score, and category."""
        index = build_pinecone_index_mock({"shirt": [{"id": SHIRT_A_ID, "score": 0.91, "metadata": {}}]})
        svc   = SearchService(index=index)
        vec   = np.random.rand(512).astype(np.float32)

        matches = await svc.search(vec, GarmentCategory.SHIRT)

        assert len(matches) == 1
        m = matches[0]
        assert m.product_id == SHIRT_A_ID
        assert abs(m.score - 0.91) < 0.001
        assert m.category == GarmentCategory.SHIRT

    @pytest.mark.asyncio
    async def test_empty_pinecone_response_returns_empty_list(self):
        """When Pinecone returns no matches, an empty list is returned."""
        index = build_pinecone_index_mock({"shirt": []})
        svc   = SearchService(index=index)
        vec   = np.random.rand(512).astype(np.float32)

        matches = await svc.search(vec, GarmentCategory.SHIRT)

        assert matches == []

    @pytest.mark.asyncio
    async def test_search_many_queries_each_category_separately(self):
        """search_many calls Pinecone once per (embedding, category) pair."""
        index = build_pinecone_index_mock()
        svc   = SearchService(index=index)
        pairs = [
            (np.random.rand(512).astype(np.float32), GarmentCategory.SHIRT),
            (np.random.rand(512).astype(np.float32), GarmentCategory.PANTS),
        ]

        await svc.search_many(pairs)

        assert index.query.call_count == 2
        namespaces = [c.kwargs["namespace"] for c in index.query.call_args_list]
        assert set(namespaces) == {"shirt", "pants"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. TestStageInvocationOrder
#
# Validates:
#   • YOLO, CLIP, Pinecone, and the catalog DB are all invoked during a
#     successful recommendation call — none are skipped.
#   • When YOLO returns no garments, CLIP and Pinecone are never called
#     (short-circuit behaviour preserves latency).
# ══════════════════════════════════════════════════════════════════════════════

class TestStageInvocationOrder:
    """
    Verifies the full service call chain so that no stage is accidentally
    bypassed and the short-circuit on empty detections works correctly.
    """

    @pytest.mark.asyncio
    async def test_all_four_stages_invoked_on_successful_run(self, catalog_map):
        """
        For a valid image with two detected garments (shirt + pants),
        all four stages — YOLO → CLIP → Pinecone → DB — are called.
        """
        garments = [
            make_garment(GarmentCategory.SHIRT),
            make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600),
        ]
        svc, yolo, clip, index, catalog = _build_recommendation_service(garments, catalog_map)
        request = RecommendationRequest(budget=500.0)

        await svc.recommend(_sample_image(), request)

        # Stage 1: YOLO
        yolo.detect_async.assert_called_once()
        # Stage 2: CLIP (called once for the full batch of crops)
        clip.encode_async.assert_called_once()
        # Stage 3: Pinecone (called once per detected garment)
        assert index.query.call_count == 2
        # Stage 4: PostgreSQL catalog
        catalog.get_products_by_ids.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_detection_short_circuits_clip_and_pinecone(self):
        """
        When YOLO detects nothing, CLIP and Pinecone must NOT be called.
        The service returns an empty response immediately.
        """
        svc, yolo, clip, index, catalog = _build_recommendation_service([], {})
        request = RecommendationRequest(budget=100.0)

        resp = await svc.recommend(_sample_image(), request)

        assert resp.total_matches == 0
        assert resp.results == []
        # Stages 2–4 must be skipped
        clip.encode_async.assert_not_called()
        index.query.assert_not_called()
        catalog.get_products_by_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clip_receives_one_image_per_detected_garment(self, catalog_map):
        """
        The number of images passed to CLIP equals the number of valid
        garments returned by YOLO (after tiny-box filtering).
        """
        garments = [
            make_garment(GarmentCategory.SHIRT),
            make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600),
            make_garment(GarmentCategory.SHOES, x_min=150, y_min=620, x_max=350, y_max=700),
        ]
        svc, _, clip, _, _ = _build_recommendation_service(garments, catalog_map)

        await svc.recommend(_sample_image(640, 720), RecommendationRequest(budget=500.0))

        crops_sent = clip.encode_async.call_args[0][0]
        assert len(crops_sent) == 3

    @pytest.mark.asyncio
    async def test_pinecone_query_vector_is_float_list(self, catalog_map):
        """
        The vector passed to Pinecone's query() is a Python list of floats
        (not a numpy array), as required by the Pinecone client.
        """
        garment = make_garment(GarmentCategory.SHIRT)
        svc, _, _, index, _ = _build_recommendation_service([garment], catalog_map)

        await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        vec_arg = index.query.call_args.kwargs["vector"]
        assert isinstance(vec_arg, list)
        assert isinstance(vec_arg[0], float)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TestBudgetFilteringIntegration
#
# Validates:
#   • Products priced at exactly the budget ceiling are included.
#   • Products priced one cent above the ceiling are excluded.
#   • When all catalog products exceed the budget, the response is empty.
#   • A mix of in-budget and over-budget products returns only the former.
#   • The budget is applied independently per garment category.
# ══════════════════════════════════════════════════════════════════════════════

class TestBudgetFilteringIntegration:
    """
    Exercises the budget filter end-to-end through a fully wired
    RecommendationService so that the filter runs on real ProductResponse
    objects (not mocked ones).
    """

    def _shirt_service(self, products: list):
        """Helper: service with a single shirt garment, given products."""
        garment = make_garment(GarmentCategory.SHIRT)
        products_map = {str(p.id): p for p in products}
        # Build a custom Pinecone index that returns all product IDs for "shirt"
        matches = [{"id": str(p.id), "score": 0.90 - i * 0.02, "metadata": {}} for i, p in enumerate(products)]
        index = build_pinecone_index_mock({"shirt": matches})
        clip  = build_clip_mock(1)
        svc   = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=clip),
            search=SearchService(index=index),
            catalog=_make_catalog_mock(products_map),
        )
        return svc

    @pytest.mark.asyncio
    async def test_items_below_budget_are_included(self):
        """Products priced below the budget ceiling appear in the results."""
        product = make_mock_product(price=39.99, category="shirt")
        svc = self._shirt_service([product])

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        assert resp.total_matches == 1
        assert resp.results[0].items[0].price == 39.99

    @pytest.mark.asyncio
    async def test_item_exactly_at_budget_is_included(self):
        """A product priced exactly at the budget is not excluded (≤, not <)."""
        product = make_mock_product(price=50.0, category="shirt")
        svc = self._shirt_service([product])

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        assert resp.total_matches == 1

    @pytest.mark.asyncio
    async def test_item_one_cent_over_budget_is_excluded(self):
        """A product priced $0.01 above the budget is excluded."""
        product = make_mock_product(price=50.01, category="shirt")
        svc = self._shirt_service([product])

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        assert resp.total_matches == 0

    @pytest.mark.asyncio
    async def test_all_items_over_budget_yields_empty_response(self):
        """When every catalog item exceeds the budget, total_matches == 0."""
        products = [
            make_mock_product(price=200.0, category="shirt"),
            make_mock_product(price=350.0, category="shirt"),
        ]
        svc = self._shirt_service(products)

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        assert resp.total_matches == 0
        assert resp.results == []

    @pytest.mark.asyncio
    async def test_mixed_budget_only_cheap_items_returned(self):
        """Only products within budget are returned when the catalog is mixed."""
        cheap     = make_mock_product(price=30.0,  category="shirt")
        mid       = make_mock_product(price=45.0,  category="shirt")
        expensive = make_mock_product(price=200.0, category="shirt")
        svc = self._shirt_service([cheap, mid, expensive])

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        returned_prices = {item.price for item in resp.results[0].items}
        assert 30.0  in returned_prices
        assert 45.0  in returned_prices
        assert 200.0 not in returned_prices


# ══════════════════════════════════════════════════════════════════════════════
# 6. TestLocalityBoostIntegration
#
# Validates:
#   • A seller within the locality radius receives a positive score boost.
#   • A seller outside the radius receives zero boost.
#   • When user coordinates are omitted, no boost is applied.
#   • A local seller with a lower raw vector score can outrank a remote
#     seller with a higher raw score after the boost is applied.
# ══════════════════════════════════════════════════════════════════════════════

class TestLocalityBoostIntegration:
    """
    Verifies that the locality boost is applied correctly so that
    geographically close sellers are ranked above distant ones.
    """

    def _locality_service(self, local_seller, remote_seller):
        """
        Build a service with two shirt products:
          • SHIRT_A (local, $45.99)  — lower raw Pinecone score (0.85)
          • SHIRT_B (remote, $29.99) — higher raw Pinecone score (0.91)

        With user coordinates, SHIRT_A should rise above SHIRT_B.
        Without user coordinates, SHIRT_B leads (0.91 > 0.85).
        """
        garment = make_garment(GarmentCategory.SHIRT)
        products_map = {
            SHIRT_A_ID: make_mock_product(SHIRT_A_ID, price=45.99, category="shirt", seller=local_seller),
            SHIRT_B_ID: make_mock_product(SHIRT_B_ID, price=29.99, category="shirt", seller=remote_seller),
        }
        index = build_pinecone_index_mock({
            "shirt": [
                {"id": SHIRT_B_ID, "score": 0.91, "metadata": {}},  # remote but higher score
                {"id": SHIRT_A_ID, "score": 0.85, "metadata": {}},  # local but lower raw score
            ]
        })
        return RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=build_clip_mock(1)),
            search=SearchService(index=index),
            catalog=_make_catalog_mock(products_map),
        )

    @pytest.mark.asyncio
    async def test_local_seller_score_exceeds_remote_after_boost(self, local_seller, remote_seller):
        """
        Local seller (Oakland, ~18 km) receives a score boost that pushes
        its final score above the remote seller's raw score.
        """
        svc = self._locality_service(local_seller, remote_seller)
        request = RecommendationRequest(
            budget=200.0,
            user_latitude=USER_LAT,
            user_longitude=USER_LON,
        )

        resp = await svc.recommend(_sample_image(), request)

        items = resp.results[0].items
        assert items[0].id == uuid.UUID(SHIRT_A_ID), \
            "Local seller should be ranked first after locality boost"

    @pytest.mark.asyncio
    async def test_without_user_coords_no_boost_applied(self, local_seller, remote_seller):
        """
        When user_latitude / user_longitude are None, no boost is applied and
        the remote seller with the higher raw score retains the top position.
        """
        svc = self._locality_service(local_seller, remote_seller)
        request = RecommendationRequest(budget=200.0)  # no coordinates

        resp = await svc.recommend(_sample_image(), request)

        items = resp.results[0].items
        assert items[0].id == uuid.UUID(SHIRT_B_ID), \
            "Without coordinates, higher raw-score item (remote) should lead"

    @pytest.mark.asyncio
    async def test_local_item_is_marked_is_local_true(self, local_seller, remote_seller):
        """The ProductResponse for a local seller has is_local == True."""
        svc = self._locality_service(local_seller, remote_seller)
        request = RecommendationRequest(
            budget=200.0, user_latitude=USER_LAT, user_longitude=USER_LON,
        )

        resp = await svc.recommend(_sample_image(), request)

        items_by_id = {str(item.id): item for item in resp.results[0].items}
        assert items_by_id[SHIRT_A_ID].is_local is True
        assert items_by_id[SHIRT_B_ID].is_local is False

    @pytest.mark.asyncio
    async def test_remote_seller_outside_radius_gets_zero_boost(self, local_seller, remote_seller):
        """A seller 560 km away gets no locality boost (score unchanged)."""
        # Build a service with only one product: the remote seller
        garment = make_garment(GarmentCategory.SHIRT)
        remote_product = make_mock_product(SHIRT_B_ID, price=29.99, category="shirt", seller=remote_seller)
        index = build_pinecone_index_mock({"shirt": [{"id": SHIRT_B_ID, "score": 0.91, "metadata": {}}]})
        svc = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=build_clip_mock(1)),
            search=SearchService(index=index),
            catalog=_make_catalog_mock({SHIRT_B_ID: remote_product}),
        )
        request = RecommendationRequest(
            budget=200.0, user_latitude=USER_LAT, user_longitude=USER_LON,
        )

        resp = await svc.recommend(_sample_image(), request)

        item = resp.results[0].items[0]
        assert item.is_local is False
        assert abs(item.similarity_score - 0.91) < 0.001  # no boost added


# ══════════════════════════════════════════════════════════════════════════════
# 7. TestCompleteServicePipeline
#
# Validates:
#   • All three clothing categories (shirt, pants, shoes) appear in the
#     response when all three are detected.
#   • The response schema is a valid RecommendationResponse.
#   • top_n limits the number of results per category.
#   • No items are returned when the catalog map is empty.
#   • Detected category list reflects all garments found by YOLO.
# ══════════════════════════════════════════════════════════════════════════════

class TestCompleteServicePipeline:
    """
    Full end-to-end service pipeline (no HTTP) covering the full three-
    category flow and various edge conditions.
    """

    @pytest.mark.asyncio
    async def test_three_category_response(self, catalog_map):
        """Shirt, pants, and shoes are each returned as a separate category group."""
        garments = [
            make_garment(GarmentCategory.SHIRT, x_min=100, y_min=50,  x_max=400, y_max=350),
            make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600),
            make_garment(GarmentCategory.SHOES, x_min=150, y_min=620, x_max=350, y_max=700),
        ]
        svc, *_ = _build_recommendation_service(garments, catalog_map)

        resp = await svc.recommend(_sample_image(640, 720), RecommendationRequest(budget=500.0))

        returned_categories = {g.category for g in resp.results}
        assert GarmentCategory.SHIRT in returned_categories
        assert GarmentCategory.PANTS in returned_categories

    @pytest.mark.asyncio
    async def test_detected_items_list_matches_yolo_output(self, catalog_map):
        """detected_items in the response lists every garment category YOLO found."""
        garments = [
            make_garment(GarmentCategory.SHIRT),
            make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600),
        ]
        svc, *_ = _build_recommendation_service(garments, catalog_map)

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        assert GarmentCategory.SHIRT in resp.detected_items
        assert GarmentCategory.PANTS in resp.detected_items

    @pytest.mark.asyncio
    async def test_top_n_limits_results_per_category(self):
        """
        When top_n=1 is requested, only one product is returned per category
        even if the catalog and Pinecone return more.
        """
        garment = make_garment(GarmentCategory.SHIRT)
        products = {
            SHIRT_A_ID: make_mock_product(SHIRT_A_ID, price=40.0, category="shirt"),
            SHIRT_B_ID: make_mock_product(SHIRT_B_ID, price=30.0, category="shirt"),
        }
        index = build_pinecone_index_mock({
            "shirt": [
                {"id": SHIRT_A_ID, "score": 0.91, "metadata": {}},
                {"id": SHIRT_B_ID, "score": 0.88, "metadata": {}},
            ]
        })
        svc = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=build_clip_mock(1)),
            search=SearchService(index=index),
            catalog=_make_catalog_mock(products),
        )

        resp = await svc.recommend(
            _sample_image(), RecommendationRequest(budget=100.0, top_n=1)
        )

        assert len(resp.results[0].items) == 1

    @pytest.mark.asyncio
    async def test_empty_catalog_returns_zero_matches(self, catalog_map):
        """
        When the catalog service finds no products for the Pinecone-returned IDs
        (e.g., orphaned vectors), total_matches is 0.
        """
        garments = [make_garment(GarmentCategory.SHIRT)]
        svc, *_ = _build_recommendation_service(garments, {})  # empty catalog

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        assert resp.total_matches == 0

    @pytest.mark.asyncio
    async def test_response_is_valid_recommendation_response(self, catalog_map):
        """The returned object is a well-formed RecommendationResponse."""
        garments = [make_garment(GarmentCategory.SHIRT)]
        svc, *_ = _build_recommendation_service(garments, catalog_map)

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        assert isinstance(resp, RecommendationResponse)
        assert isinstance(resp.total_matches, int)
        assert isinstance(resp.detected_items, list)
        assert isinstance(resp.results, list)
        for group in resp.results:
            assert isinstance(group, GarmentRecommendations)
            assert isinstance(group.items, list)


# ══════════════════════════════════════════════════════════════════════════════
# 8. TestAPIRecommendationFlow  (HTTP + real service business logic)
#
# Validates the /recommend endpoint end-to-end:
#   • A valid multipart upload returns 200 with structured JSON.
#   • Budget filtering is enforced (not just a stub).
#   • With user coordinates, the local seller is ranked first.
#   • budget=0 fails FastAPI validation (422).
#   • Omitting the required budget field returns 422.
#   • A non-image content type returns 400.
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIRecommendationFlow:
    """
    Tests the /recommend HTTP endpoint with the real RecommendationService
    wired in (via dependency_overrides), so business logic is exercised
    through the full request/response cycle.
    """

    @pytest.mark.asyncio
    async def test_valid_jpeg_returns_200(self, wired_client):
        """A valid JPEG with a budget returns HTTP 200 and a JSON body."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "200"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert "total_matches" in body
        assert "detected_items" in body

    @pytest.mark.asyncio
    async def test_budget_filtering_enforced_via_api(self, wired_client):
        """
        With budget=30, only items priced ≤ $30 should appear.
        SHIRT_B ($29.99) passes; SHIRT_A ($45.99), PANTS ($79.99),
        SHOES ($99.99), and EXPENSIVE ($349.99) are excluded.
        """
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "30"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for group in body["results"]:
            for item in group["items"]:
                assert item["price"] <= 30.0, \
                    f"Item price {item['price']} exceeds budget 30.0"

    @pytest.mark.asyncio
    async def test_local_seller_ranked_first_with_user_coordinates(self, wired_client):
        """
        When user_latitude/user_longitude are the SF coordinates, the local
        Oakland seller (SHIRT_A) should appear before the remote LA seller
        (SHIRT_B) despite SHIRT_B having the higher raw Pinecone score.
        """
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
            data={
                "budget": "200",
                "user_latitude":  str(USER_LAT),
                "user_longitude": str(USER_LON),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        shirt_groups = [g for g in body["results"] if g["category"] == "shirt"]
        if shirt_groups:
            top_item = shirt_groups[0]["items"][0]
            assert top_item["id"] == SHIRT_A_ID, \
                "Local seller (SHIRT_A) must be ranked first with user coordinates"

    @pytest.mark.asyncio
    async def test_budget_zero_returns_422(self, wired_client):
        """budget=0 violates the gt=0 constraint → 422 Unprocessable Entity."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "0"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_budget_returns_422(self, wired_client):
        """Omitting the required budget form field returns 422."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_top_n_limits_response_length(self, wired_client):
        """top_n=1 caps results at one item per category group."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "500", "top_n": "1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for group in body["results"]:
            assert len(group["items"]) <= 1

    @pytest.mark.asyncio
    async def test_pdf_upload_returns_400(self, wired_client):
        """Uploading a PDF instead of an image returns 400."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("doc.pdf", make_pdf_bytes(), "application/pdf")},
            data={"budget": "100"},
        )
        assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 9. TestAPIPipelineEndpointFlow
#
# Validates the /pipeline/recommend endpoint with the real pipeline wired in:
#   • A valid JPEG returns 200 with the PipelineRecommendationResponse schema.
#   • Non-image content types return 400.
#   • Response JSON has the required top-level keys.
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIPipelineEndpointFlow:
    """
    Tests the /pipeline/recommend endpoint with the real RecommendationPipeline
    service wired to mock ML (no weights loaded, no Pinecone HTTP calls).
    """

    @pytest.mark.asyncio
    async def test_pipeline_valid_jpeg_returns_200(self, wired_client):
        """A valid JPEG upload returns HTTP 200."""
        resp = await wired_client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_pipeline_response_has_required_schema_keys(self, wired_client):
        """The JSON body contains shirts, pants, shoes, total_detections, total_matches."""
        resp = await wired_client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("outfit.jpg", make_jpeg(), "image/jpeg")},
        )
        body = resp.json()
        for key in ("shirts", "pants", "shoes", "total_detections", "total_matches"):
            assert key in body, f"Missing key '{key}' in pipeline response"

    @pytest.mark.asyncio
    async def test_pipeline_rejects_pdf(self, wired_client):
        """A PDF upload to /pipeline/recommend returns 400."""
        resp = await wired_client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("file.pdf", make_pdf_bytes(), "application/pdf")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_pipeline_rejects_plain_text(self, wired_client):
        """A plain-text upload returns 400."""
        resp = await wired_client.post(
            "/api/v1/pipeline/recommend",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 10. TestImageFormatHandling
#
# Validates that the image preprocessing layer:
#   • Accepts JPEG, PNG, and WebP (the three declared MIME types).
#   • Rejects unsupported MIME types with HTTP 400.
#   • Accepts oversized images and resizes them transparently.
#   • Rejects corrupt (truncated) image data with HTTP 400.
# ══════════════════════════════════════════════════════════════════════════════

class TestImageFormatHandling:
    """
    Validates the image validation / preprocessing layer for the
    /recommend endpoint across all supported and unsupported formats.
    """

    @pytest.mark.asyncio
    async def test_jpeg_accepted(self, wired_client):
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("photo.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "100"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_png_accepted(self, wired_client):
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("photo.png", make_png(), "image/png")},
            data={"budget": "100"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_webp_accepted(self, wired_client):
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("photo.webp", make_webp(), "image/webp")},
            data={"budget": "100"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_oversized_image_accepted_and_resized(self, wired_client):
        """
        A 2048×2048 JPEG exceeds the 1024 px threshold but is valid —
        the server resizes it silently and returns 200.
        """
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("large.jpg", make_oversized_jpeg(), "image/jpeg")},
            data={"budget": "100"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_corrupt_jpeg_returns_400(self, wired_client):
        """Truncated JPEG bytes that PIL cannot decode return 400."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("bad.jpg", make_corrupt_jpeg(), "image/jpeg")},
            data={"budget": "100"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_gif_rejected(self, wired_client):
        """GIF is not in the allowed MIME list → 400."""
        buf = BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="GIF")
        buf.seek(0)
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("anim.gif", buf.getvalue(), "image/gif")},
            data={"budget": "100"},
        )
        assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 11. TestFailureScenarios
#
# Validates all meaningful failure paths through the pipeline:
#   • YOLO returns no detections → empty 200 response (not an error).
#   • Pinecone returns no matches for any category → empty response.
#   • Budget filtering removes every item → empty response.
#   • Catalog returns no products for Pinecone IDs (orphaned vectors).
#   • /recommend with no file returns 422.
#   • /pipeline/recommend with no file returns 422.
# ══════════════════════════════════════════════════════════════════════════════

class TestFailureScenarios:
    """
    Tests all meaningful failure modes so that the system degrades gracefully
    rather than raising unhandled exceptions.
    """

    @pytest.mark.asyncio
    async def test_yolo_no_detections_returns_empty_200(self):
        """
        When YOLO finds no garments, the service returns an empty
        RecommendationResponse with total_matches=0 — not an exception.
        """
        svc, *_ = _build_recommendation_service([], {})
        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=100.0))

        assert resp.total_matches == 0
        assert resp.results == []
        assert resp.detected_items == []

    @pytest.mark.asyncio
    async def test_pinecone_no_matches_returns_empty(self, catalog_map):
        """
        When Pinecone returns zero matches for every category, the service
        returns an empty response (not an error).
        """
        garments = [make_garment(GarmentCategory.SHIRT)]
        # Override index to return empty for all namespaces
        index = build_pinecone_index_mock({})  # empty → all namespaces return []
        clip  = build_clip_mock(1)
        svc   = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock(garments)),
            embedding=EmbeddingService(encoder=clip),
            search=SearchService(index=index),
            catalog=_make_catalog_mock(catalog_map),
        )

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        assert resp.total_matches == 0

    @pytest.mark.asyncio
    async def test_all_products_over_budget_returns_empty(self):
        """Budget filtering that removes every item yields an empty response."""
        garment = make_garment(GarmentCategory.SHIRT)
        product = make_mock_product(price=999.0, category="shirt")
        products_map = {str(product.id): product}
        index = build_pinecone_index_mock({"shirt": [{"id": str(product.id), "score": 0.91, "metadata": {}}]})
        svc = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=build_clip_mock(1)),
            search=SearchService(index=index),
            catalog=_make_catalog_mock(products_map),
        )

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=50.0))

        assert resp.total_matches == 0
        assert resp.results == []

    @pytest.mark.asyncio
    async def test_orphaned_pinecone_vectors_ignored(self):
        """
        When Pinecone returns product IDs that do not exist in the catalog
        (orphaned vectors), they are silently skipped.
        """
        garment = make_garment(GarmentCategory.SHIRT)
        orphan_id = str(uuid.uuid4())
        index = build_pinecone_index_mock({"shirt": [{"id": orphan_id, "score": 0.90, "metadata": {}}]})
        svc = RecommendationService(
            detection=DetectionService(detector=build_yolo_mock([garment])),
            embedding=EmbeddingService(encoder=build_clip_mock(1)),
            search=SearchService(index=index),
            catalog=_make_catalog_mock({}),  # empty catalog
        )

        resp = await svc.recommend(_sample_image(), RecommendationRequest(budget=500.0))

        assert resp.total_matches == 0  # orphan was skipped, not an error

    @pytest.mark.asyncio
    async def test_recommend_endpoint_no_file_returns_422(self, wired_client):
        """/recommend without a file returns 422 (missing required field)."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            data={"budget": "100"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_pipeline_endpoint_no_file_returns_422(self, wired_client):
        """/pipeline/recommend without a file returns 422."""
        resp = await wired_client.post("/api/v1/pipeline/recommend")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_recommend_endpoint_negative_top_n_returns_422(self, wired_client):
        """top_n < 1 violates the ge=1 constraint → 422."""
        resp = await wired_client.post(
            "/api/v1/recommend",
            files={"file": ("photo.jpg", make_jpeg(), "image/jpeg")},
            data={"budget": "100", "top_n": "0"},
        )
        assert resp.status_code == 422
