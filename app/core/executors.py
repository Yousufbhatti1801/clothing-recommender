"""Dedicated thread-pool executor for CPU-bound ML inference.

Using a separate executor keeps ML tasks from competing for the same
thread pool slots as asyncio I/O callbacks and the default executor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

# Two workers: one for CLIP, one for YOLO — they rarely overlap in practice.
# Increase to 4 if you run on a multi-core VM and want higher concurrency.
_ML_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="ml-worker",
)


def get_ml_executor() -> ThreadPoolExecutor:
    """Return the process-wide ML thread-pool executor."""
    return _ML_EXECUTOR


def shutdown_ml_executor(wait: bool = True) -> None:
    """Gracefully shut down the ML executor (called during app shutdown)."""
    _ML_EXECUTOR.shutdown(wait=wait)
