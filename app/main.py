"""FastAPI application factory with startup/shutdown lifespan."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.core.config import get_settings
from app.core.database import engine
from app.core.pinecone_client import init_pinecone
from app.models.orm import Base
from ml.clip_encoder import get_clip_encoder
from ml.yolo_detector import get_yolo_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:  create DB tables, warm up ML models, connect to Pinecone.
    Shutdown: dispose DB engine.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    init_pinecone()

    # Pre-load singleton ML models (avoids cold-start on first request)
    get_clip_encoder()
    get_yolo_detector()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="AI-powered fashion recommendation API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    return app


app = create_app()
