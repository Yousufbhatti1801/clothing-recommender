"""Retry decorator with exponential backoff for external service calls."""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    backoff: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Synchronous retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including the first try).
        backoff:      Base wait time in seconds; doubles on each retry.
        exceptions:   Only retry on these exception types.

    Usage::

        @with_retry(max_attempts=3, backoff=0.5)
        def query_pinecone(...):
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts,
                            fn.__qualname__,
                            exc,
                        )
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    log.warning(
                        "Attempt %d/%d failed for %s (%s). Retrying in %.2fs…",
                        attempt,
                        max_attempts,
                        fn.__qualname__,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            raise last_exc  # unreachable but satisfies type checkers
        return wrapper
    return decorator
