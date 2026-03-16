"""Shared httpx AsyncClient — created once in lifespan, reused everywhere."""
from __future__ import annotations

import httpx

# Process-wide client; initialized by the FastAPI lifespan hook.
_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """
    Return the shared ``httpx.AsyncClient``.

    Raises:
        RuntimeError if called before the lifespan startup has run.
    """
    if _client is None:
        raise RuntimeError(
            "HTTP client is not initialized. "
            "Ensure the FastAPI lifespan hook has run before making requests."
        )
    return _client


def _set_http_client(client: httpx.AsyncClient) -> None:
    """Internal: called by the lifespan hook to register the shared client."""
    global _client
    _client = client


def _clear_http_client() -> None:
    """Internal: called by the lifespan hook on shutdown."""
    global _client
    _client = None
