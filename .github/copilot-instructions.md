# Copilot Instructions

## Project goal
Build an AI fashion recommendation app:
- user uploads an image
- detect shirt, pants, shoes separately
- generate embeddings
- search similar catalog items
- filter by user budget
- prefer local sellers when possible

## Tech stack
- Python backend
- FastAPI
- PostgreSQL
- Pinecone
- CLIP or similar vision embedding model

## Build/Test
- backend run: uvicorn app:app --reload
- tests: pytest
- lint: ruff check .