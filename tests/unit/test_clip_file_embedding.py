"""Unit tests for ClipFileEmbeddingService — mocked CLIP encoder."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.clip_file_embedding import ClipFileEmbeddingService


@pytest.fixture
def tmp_image_path(tmp_path) -> str:
    """Create a real JPEG on disk for file-based embedding tests."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    path = tmp_path / "test_shoe.jpg"
    img.save(path, format="JPEG")
    return str(path)


@pytest.fixture
def mock_encoder():
    encoder = MagicMock()

    def _encode(images):
        return np.random.rand(len(images), 512).astype(np.float32)

    encoder.encode = MagicMock(side_effect=_encode)
    return encoder


class TestEmbedPath:
    def test_single_file(self, tmp_image_path, mock_encoder):
        svc = ClipFileEmbeddingService(encoder=mock_encoder)
        vec = svc.embed_path(tmp_image_path)
        assert vec.shape == (512,)
        assert vec.dtype == np.float32

    def test_file_not_found(self, mock_encoder):
        svc = ClipFileEmbeddingService(encoder=mock_encoder)
        with pytest.raises(FileNotFoundError):
            svc.embed_path("/nonexistent/image.jpg")


class TestEmbedPaths:
    def test_multiple_files(self, tmp_path, mock_encoder):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            Image.new("RGB", (224, 224)).save(p)
            paths.append(str(p))

        svc = ClipFileEmbeddingService(encoder=mock_encoder)
        results = svc.embed_paths(paths)
        assert len(results) == 3
        for v in results:
            assert v.shape == (512,)

    def test_empty_list(self, mock_encoder):
        svc = ClipFileEmbeddingService(encoder=mock_encoder)
        results = svc.embed_paths([])
        assert results == []
        mock_encoder.encode.assert_not_called()
