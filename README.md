# AI Clothing Recommender

An AI-powered fashion recommendation API built with FastAPI, CLIP, YOLOv8, Pinecone, and PostgreSQL.

## Quickstart

```bash
# 1. Copy and fill in env vars
cp .env.example .env
# edit .env — set PINECONE_API_KEY and DB_URL at minimum

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL (or use Docker)
docker compose up db -d

# 4. Seed the database + Pinecone vector index  ← NEW
python3 scripts/seed_catalog.py
# Inserts 5 sellers, 39 products, CLIP-embeds each image, and upserts
# vectors with rich metadata (price, brand, category, seller_id) to Pinecone.
# Run with --reset to wipe and re-seed, or --dry-run to verify image URLs.

# 5. Run the API
uvicorn app.main:app --reload
```

## Docker (full stack)

```bash
docker compose up --build
```

API will be available at http://localhost:8000  
Interactive docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/pipeline/recommend` | Upload image → detect garments → embed → search Pinecone → return matches |
| `POST` | `/api/v1/recommend` | Upload outfit image, get recommendations |
| `POST` | `/api/v1/catalog/ingest` | Index a new product into PostgreSQL + Pinecone |
| `GET`  | `/api/v1/health` | Liveness + readiness check |

### Example: Pipeline Recommendation (full AI flow)

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/recommend \
  -F "file=@outfit.jpg" \
  -F "budget=150"
# Returns detected garments + top similar catalog products under $150.
```

### Example: Recommendation with locality boost

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -F "file=@outfit.jpg" \
  -F "budget=150" \
  -F "user_latitude=40.7128" \
  -F "user_longitude=-74.0060" \
  -F "top_n=5"
```

## Testing

```bash
pytest
```

## Linting

```bash
ruff check .
```

## Project Structure

```
clothing-recommender/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── api/routes/              # HTTP route handlers
│   ├── services/                # Business logic & AI pipeline
│   ├── models/                  # ORM + Pydantic schemas
│   ├── core/                    # Config, DB, Pinecone client, DI
│   └── utils/                   # Image processing, geospatial
├── ml/
│   ├── clip_encoder.py          # CLIP singleton wrapper
│   ├── yolo_detector.py         # YOLOv8 singleton wrapper
│   └── models/                  # Model weight files (git-ignored)
├── db/
│   └── seed.py                  # Sample data seeder
├── tests/                       # Pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
