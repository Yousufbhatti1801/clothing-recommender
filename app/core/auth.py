"""API key authentication dependency for FastAPI routes."""
from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


async def require_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """
    FastAPI dependency — validates the ``X-API-Key`` request header.

    Raises:
        HTTPException 403 if the key is missing or does not match the
        configured ``API_KEY`` setting.
    """
    settings = get_settings()
    if not secrets_equal(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key


def secrets_equal(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    import hmac
    return hmac.compare_digest(a.encode(), b.encode())
