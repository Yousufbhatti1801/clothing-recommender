from pinecone import Pinecone, Index

from app.core.config import get_settings

_pinecone_client: Pinecone | None = None
_index: Index | None = None


def init_pinecone() -> None:
    global _pinecone_client, _index
    settings = get_settings()
    _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    _index = _pinecone_client.Index(settings.pinecone_index_name)


def get_pinecone_index() -> Index:
    if _index is None:
        raise RuntimeError("Pinecone has not been initialised. Call init_pinecone() first.")
    return _index
