"""Unit tests for CatalogService — mocked AsyncSession."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import GarmentCategory, ProductIngestRequest
from app.services.catalog import CatalogService
from tests.conftest import make_mock_product


def _mock_session(products: list | None = None):
    """Build a MagicMock AsyncSession that returns *products* on execute."""
    session = AsyncMock()

    if products is not None:
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = products
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute.return_value = result_mock
    else:
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute.return_value = result_mock

    return session


class TestGetProductsByIds:
    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_id(self):
        pid = str(uuid.uuid4())
        product = make_mock_product(product_id=pid)
        session = _mock_session([product])

        svc = CatalogService(db=session)
        result = await svc.get_products_by_ids([pid])
        assert pid in result
        assert result[pid].name == "Test Item"

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty_dict(self):
        session = _mock_session()
        svc = CatalogService(db=session)
        result = await svc.get_products_by_ids([])
        assert result == {}
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_products(self):
        pids = [str(uuid.uuid4()) for _ in range(3)]
        products = [make_mock_product(product_id=pid) for pid in pids]
        session = _mock_session(products)

        svc = CatalogService(db=session)
        result = await svc.get_products_by_ids(pids)
        assert len(result) == 3


class TestCreateProduct:
    @pytest.mark.asyncio
    async def test_creates_and_flushes(self):
        session = AsyncMock()
        session.add = MagicMock()  # add() is synchronous in SQLAlchemy
        svc = CatalogService(db=session)

        data = ProductIngestRequest(
            name="Blue Polo",
            category=GarmentCategory.SHIRT,
            price=29.99,
            image_url="https://example.com/polo.jpg",
        )
        await svc.create_product(data)
        session.add.assert_called_once()
        session.flush.assert_awaited_once()


class TestGetSeller:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute.return_value = result_mock

        svc = CatalogService(db=session)
        seller = await svc.get_seller(uuid.uuid4())
        assert seller is None
