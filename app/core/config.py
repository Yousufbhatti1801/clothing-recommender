from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_name: str = "AI Clothing Recommender"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Security
    api_key: str = "change-me-in-production"  # Override via API_KEY env var
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    rate_limit_per_minute: int = 20

    # PostgreSQL
    db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/clothing_db"

    # Pinecone
    pinecone_api_key: str = "change-me"
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "clothing-embeddings"

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"  # "cuda" for GPU

    # YOLOv8
    yolo_model_path: str = "ml/models/yolov8_fashion.pt"
    yolo_confidence_threshold: float = 0.4

    # Search
    pinecone_top_k: int = 20

    # Locality boost
    locality_boost: float = 0.15
    locality_radius_km: float = 50.0

    # Observability
    sentry_dsn: str | None = None

    # Ingestion safety — comma-separated allowed image domains.
    # Leave empty in development to allow all hosts.
    # Example: "cdn.mystore.com,images.mystore.com"
    allowed_image_hosts: list[str] = []

    @field_validator("allowed_image_hosts", mode="before")
    @classmethod
    def parse_hosts(cls, v: object) -> list[str]:
        """Accept a comma-separated string from env vars."""
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v  # already a list


@lru_cache
def get_settings() -> Settings:
    return Settings()
