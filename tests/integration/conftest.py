"""
tests/integration/conftest.py
==============================
Shared fixtures for the integration test suite.

Scope notes
-----------
• ``module``-scoped fixtures (catalog, sellers, image bytes) are cheap to
  create and safe to share — their contents never mutate between tests.
• ``function``-scoped mock objects (YOLO, CLIP, Pinecone) are recreated for
  every test so call-count assertions are always clean.
"""
from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image

from app.models.schemas import GarmentCategory
from tests.conftest import make_garment, make_mock_product, make_mock_seller
from tests.fixtures.images import (
    make_jpeg,
    make_outfit_jpeg,
    make_oversized_jpeg,
    make_png,
    make_webp,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixed product UUIDs
# (same string used in both Pinecone mock returns and catalog mock returns)
# ══════════════════════════════════════════════════════════════════════════════

SHIRT_A_ID   = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"  # $45.99, local seller
SHIRT_B_ID   = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"  # $29.99, remote seller
PANTS_A_ID   = "cccccccc-cccc-cccc-cccc-cccccccccccc"  # $79.99, remote seller
SHOES_A_ID   = "dddddddd-dddd-dddd-dddd-dddddddddddd"  # $99.99, no seller
EXPENSIVE_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"  # $349.99, remote seller

# ══════════════════════════════════════════════════════════════════════════════
# Geolocation constants (used by locality-boost tests)
# ══════════════════════════════════════════════════════════════════════════════

# Simulated user position
USER_LAT, USER_LON = 37.7749, -122.4194           # San Francisco

# Local seller: Oakland — ~18 km from SF (inside 50 km radius)
LOCAL_LAT, LOCAL_LON = 37.8044, -122.2712

# Remote seller: Los Angeles — ~560 km from SF (outside 50 km radius)
REMOTE_LAT, REMOTE_LON = 34.0522, -118.2437

# ══════════════════════════════════════════════════════════════════════════════
# Image byte fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def outfit_jpeg() -> bytes:
    """Standard 640×480 JPEG outfit image."""
    return make_jpeg(640, 480)


@pytest.fixture(scope="module")
def outfit_jpeg_multiband() -> bytes:
    """640×960 JPEG with distinct colour bands per clothing zone."""
    return make_outfit_jpeg()


@pytest.fixture(scope="module")
def outfit_png() -> bytes:
    """320×480 PNG variant."""
    return make_png(320, 480)


@pytest.fixture(scope="module")
def outfit_webp() -> bytes:
    """400×600 WebP variant."""
    return make_webp(400, 600)


@pytest.fixture(scope="module")
def oversized_jpeg() -> bytes:
    """2048×2048 JPEG — exercises the resize-before-inference path."""
    return make_oversized_jpeg()


# ══════════════════════════════════════════════════════════════════════════════
# Seller & product catalog (module-scoped, immutable)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def local_seller():
    """Seller inside the 50 km locality radius from San Francisco."""
    return make_mock_seller(
        seller_id=str(uuid.uuid4()),
        name="Oakland Threads",
        city="Oakland",
        country="US",
        latitude=LOCAL_LAT,
        longitude=LOCAL_LON,
    )


@pytest.fixture(scope="module")
def remote_seller():
    """Seller outside the 50 km locality radius (Los Angeles)."""
    return make_mock_seller(
        seller_id=str(uuid.uuid4()),
        name="LA Fashion House",
        city="Los Angeles",
        country="US",
        latitude=REMOTE_LAT,
        longitude=REMOTE_LON,
    )


@pytest.fixture(scope="module")
def catalog_map(local_seller, remote_seller):
    """
    Full test product catalog keyed by str(uuid).

    Prices span the budget test range:
      $29.99  SHIRT_B   — cheapest shirt (remote seller)
      $45.99  SHIRT_A   — mid-price shirt (local seller)
      $79.99  PANTS_A   — mid-price pants (remote seller)
      $99.99  SHOES_A   — shoes (no seller)
      $349.99 EXPENSIVE — luxury shirt (remote seller)
    """
    return {
        SHIRT_A_ID:   make_mock_product(SHIRT_A_ID,   "Blue Polo",       "UrbanCo",  "shirt", price=45.99,  seller=local_seller),
        SHIRT_B_ID:   make_mock_product(SHIRT_B_ID,   "Red Tee",         "Basics",   "shirt", price=29.99,  seller=remote_seller),
        PANTS_A_ID:   make_mock_product(PANTS_A_ID,   "Slim Jeans",      "DenimLab", "pants", price=79.99,  seller=remote_seller),
        SHOES_A_ID:   make_mock_product(SHOES_A_ID,   "White Sneakers",  "StepUp",   "shoes", price=99.99,  seller=None),
        EXPENSIVE_ID: make_mock_product(EXPENSIVE_ID, "Designer Blazer", "LuxBrand", "shirt", price=349.99, seller=remote_seller),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pinecone namespace → match data
#
# Pinecone scores are INTENTIONALLY ordered so that the remote-seller shirt
# (SHIRT_B, score 0.91) beats the local-seller shirt (SHIRT_A, score 0.85)
# in raw vector similarity.  Locality boost then reverses that order.
# ══════════════════════════════════════════════════════════════════════════════

PINECONE_MATCHES: dict[str, list[dict]] = {
    "shirt": [
        {"id": SHIRT_B_ID,   "score": 0.91, "metadata": {}},
        {"id": SHIRT_A_ID,   "score": 0.85, "metadata": {}},
        {"id": EXPENSIVE_ID, "score": 0.80, "metadata": {}},
    ],
    "pants": [
        {"id": PANTS_A_ID, "score": 0.87, "metadata": {}},
    ],
    "shoes": [
        {"id": SHOES_A_ID, "score": 0.82, "metadata": {}},
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# Function-scoped mock factories
# (call-count state must be fresh per test)
# ══════════════════════════════════════════════════════════════════════════════

def build_yolo_mock(garments) -> MagicMock:
    """Return a YOLODetector mock that always yields *garments*."""
    m = MagicMock()
    m.detect_async         = AsyncMock(return_value=garments)
    m.detect_targets_async = AsyncMock(return_value=garments)
    return m


def build_clip_mock(max_batch: int = 8) -> MagicMock:
    """Return a CLIPEncoder mock with deterministic 512-d vectors."""
    rng = np.random.default_rng(42)
    pool = rng.random((max_batch, 512)).astype(np.float32)

    m = MagicMock()

    async def _encode(images):
        n = len(images)
        if n > max_batch:
            return rng.random((n, 512)).astype(np.float32)
        return pool[:n]

    m.encode_async = AsyncMock(side_effect=_encode)
    return m


def build_pinecone_index_mock(
    matches_by_ns: dict[str, list[dict]] | None = None,
) -> MagicMock:
    """
    Return a mock Pinecone Index.

    ``matches_by_ns`` maps a Pinecone namespace string to a list of match
    dicts ``{"id": str, "score": float, "metadata": dict}``.
    Queries for unknown namespaces return an empty match list.
    Pass ``{}`` explicitly to simulate a Pinecone index with no matches.
    """
    # Use `is None` so that an explicit empty dict {} is respected
    data = PINECONE_MATCHES if matches_by_ns is None else matches_by_ns

    index = MagicMock()

    def _query(vector, top_k, namespace, include_metadata=False):  # noqa: ARG001
        hits = data.get(namespace, [])
        return {"matches": hits[:top_k]}

    index.query                = MagicMock(side_effect=_query)
    index.describe_index_stats = MagicMock(return_value={"dimension": 512})
    index.upsert               = MagicMock()
    return index


# ── Default garment set used by most tests (shirt + pants) ───────────────────

@pytest.fixture
def shirt_and_pants_garments():
    return [
        make_garment(GarmentCategory.SHIRT, x_min=100, y_min=50,  x_max=400, y_max=350, confidence=0.92),
        make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600, confidence=0.85),
    ]


@pytest.fixture
def three_category_garments():
    return [
        make_garment(GarmentCategory.SHIRT, x_min=100, y_min=50,  x_max=400, y_max=350, confidence=0.92),
        make_garment(GarmentCategory.PANTS, x_min=120, y_min=360, x_max=380, y_max=600, confidence=0.85),
        make_garment(GarmentCategory.SHOES, x_min=150, y_min=620, x_max=350, y_max=700, confidence=0.78),
    ]


@pytest.fixture
def mock_yolo(shirt_and_pants_garments):
    return build_yolo_mock(shirt_and_pants_garments)


@pytest.fixture
def mock_clip():
    return build_clip_mock()


@pytest.fixture
def mock_index():
    return build_pinecone_index_mock()
