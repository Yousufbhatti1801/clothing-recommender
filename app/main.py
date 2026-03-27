"""FastAPI application factory with startup/shutdown lifespan."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import router
from app.core.config import get_settings
from app.core.database import engine
from app.core.executors import shutdown_ml_executor
from app.core.http_client import _clear_http_client, _set_http_client
from app.core.logging_config import configure_logging
from app.core.pinecone_client import init_pinecone
from app.core.rate_limit import limiter
from app.models.orm import Base
from ml.clip_encoder import get_clip_encoder
from ml.yolo_detector import get_yolo_detector

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:  configure logging, create DB tables, warm ML models,
              connect Pinecone, initialise shared HTTP client.
    Shutdown: dispose DB engine, close HTTP client, shut down ML executor.
    """
    app.state.start_time = time.monotonic()

    settings = get_settings()
    configure_logging(debug=settings.debug, fmt=settings.log_format)

    # ── Sentry (optional) ─────────────────────────────────────────────────
    if settings.sentry_dsn:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=0.1,
            environment="production" if not settings.debug else "development",
        )
        log.info("Sentry error tracking enabled.")

    # ── Database ──────────────────────────────────────────────────────────
    log.info("Creating database tables if they don't exist…")
    from sqlalchemy import text as _sql_text
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Phase 7: add extended attribute columns to existing products tables.
        # These are no-ops (IF NOT EXISTS) on fresh installs where create_all
        # already created the columns from the updated ORM definition.
        for _stmt in [
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS colour VARCHAR(50)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS material VARCHAR(100)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS style VARCHAR(100)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS gender VARCHAR(20)",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS size_available TEXT",
            "ALTER TABLE products ADD COLUMN IF NOT EXISTS occasion VARCHAR(100)",
            "CREATE INDEX IF NOT EXISTS ix_products_category ON products (category)",
            "CREATE INDEX IF NOT EXISTS ix_products_category_price ON products (category, price)",
            "CREATE INDEX IF NOT EXISTS ix_products_colour ON products (colour)",
        ]:
            await conn.execute(_sql_text(_stmt))
    log.info("Database schema up to date.")

    # ── Pinecone ──────────────────────────────────────────────────────────
    log.info("Initialising Pinecone connection…")
    init_pinecone()

    # ── ML models (warm up before first request) ──────────────────────────
    log.info("Loading CLIP encoder…")
    get_clip_encoder()
    log.info("Loading YOLO detector…")
    get_yolo_detector()

    # ── Recommendation pipeline (lifespan-scoped, not module-level) ───────
    from app.services.recommendation_pipeline import RecommendationPipeline
    app.state.pipeline = RecommendationPipeline()
    log.info("RecommendationPipeline ready.")

    # ── Shared HTTP client ────────────────────────────────────────────────
    client = httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=False,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    _set_http_client(client)
    log.info("Shared httpx.AsyncClient ready.")

    log.info("Application startup complete.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("Shutting down…")
    await client.aclose()
    _clear_http_client()
    await engine.dispose()
    shutdown_ml_executor()
    log.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="AI-powered fashion recommendation API",
        # Disable interactive docs in production to reduce attack surface
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # ── Rate limiter ──────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # ── Request-ID middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    # ── Prometheus metrics ────────────────────────────────────────────────
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        log.info("Prometheus metrics exposed at /metrics")
    except ImportError:
        log.warning("prometheus-fastapi-instrumentator not installed; metrics disabled.")

    app.include_router(router, prefix=settings.api_prefix)

    # ── Frontend SPA Rendering ────────────────────────────────────────────
    import os
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
