from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_name: str = "AI Clothing Recommender"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # PostgreSQL
    db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/clothing_db"

    # Pinecone
    pinecone_api_key: str
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
