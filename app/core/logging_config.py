"""Structured logging configuration for the application."""
from __future__ import annotations

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    """
    Configure root logger with a structured format.

    Outputs to stdout so container runtimes (Docker, ECS, K8s) can capture
    and forward logs without any sidecar configuration.

    Format::

        2026-03-16 14:00:00,000 INFO     app.services.recommendation - Message here
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (level=%s)", logging.getLevelName(level)
    )
