import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
