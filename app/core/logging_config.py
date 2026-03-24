"""Structured logging configuration for the application."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record for cloud log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Include any extra fields attached by the caller
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__
            and k not in ("message", "asctime")
        }
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(debug: bool = False, fmt: str = "text") -> None:
    """
    Configure root logger.

    Args:
        debug: When True, set level to DEBUG.
        fmt:   ``"json"`` for structured JSON output (production / cloud);
               ``"text"`` for human-readable console output (development).
    """
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logging.basicConfig(level=level, handlers=[handler], force=True)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (level=%s, format=%s)", logging.getLevelName(level), fmt
    )
