"""Unit tests for image loading and preprocessing utilities."""
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image

from app.utils.image import (
    MAX_DIMENSION,
    MAX_IMAGE_SIZE_MB,
    SUPPORTED_CONTENT_TYPES,
    load_image_from_upload,
    resize_image,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  resize_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestResizeImage:
    def test_no_resize_when_within_limit(self, small_image):
        result = resize_image(small_image)
        assert result.size == (100, 100)

    def test_resize_large_landscape(self):
        img = Image.new("RGB", (2048, 1024))
        result = resize_image(img, max_dim=MAX_DIMENSION)
        assert max(result.size) == MAX_DIMENSION
        # aspect ratio preserved
        assert result.size[0] == MAX_DIMENSION
        assert result.size[1] == MAX_DIMENSION // 2

    def test_resize_large_portrait(self):
        img = Image.new("RGB", (512, 2048))
        result = resize_image(img, max_dim=MAX_DIMENSION)
        assert max(result.size) == MAX_DIMENSION
        assert result.size[1] == MAX_DIMENSION

    def test_resize_square(self):
        img = Image.new("RGB", (2000, 2000))
        result = resize_image(img, max_dim=MAX_DIMENSION)
        assert result.size == (MAX_DIMENSION, MAX_DIMENSION)

    def test_exact_boundary_no_resize(self):
        img = Image.new("RGB", (MAX_DIMENSION, MAX_DIMENSION))
        result = resize_image(img)
        assert result.size == (MAX_DIMENSION, MAX_DIMENSION)


# ═══════════════════════════════════════════════════════════════════════════════
#  load_image_from_upload
# ═══════════════════════════════════════════════════════════════════════════════

def _make_upload_file(content: bytes, content_type: str, filename: str = "test.jpg"):
    """Helper to build a mock UploadFile."""
    mock = MagicMock()
    mock.content_type = content_type
    mock.filename = filename
    mock.read = AsyncMock(return_value=content)
    return mock


class TestLoadImageFromUpload:
    @pytest.mark.asyncio
    async def test_valid_jpeg(self, image_bytes):
        upload = _make_upload_file(image_bytes, "image/jpeg")
        img = await load_image_from_upload(upload)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_unsupported_content_type(self):
        upload = _make_upload_file(b"fake", "application/pdf")
        with pytest.raises(HTTPException) as exc_info:
            await load_image_from_upload(upload)
        assert exc_info.value.status_code == 400
        assert "Unsupported image type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_oversized_file(self):
        big = b"\x00" * (MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
        upload = _make_upload_file(big, "image/jpeg")
        with pytest.raises(HTTPException) as exc_info:
            await load_image_from_upload(upload)
        assert exc_info.value.status_code == 400
        assert "limit" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_corrupted_image_data(self):
        upload = _make_upload_file(b"not-an-image", "image/jpeg")
        with pytest.raises(HTTPException) as exc_info:
            await load_image_from_upload(upload)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_supported_content_types_all_accepted(self, image_bytes):
        for ct in SUPPORTED_CONTENT_TYPES:
            # Use JPEG bytes for all — the PIL parse may fail for png/webp but
            # we mainly test that the content-type filter passes.
            upload = _make_upload_file(image_bytes, ct)
            img = await load_image_from_upload(upload)
            assert isinstance(img, Image.Image)
