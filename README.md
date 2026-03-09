# AI Clothing Recommender

An AI-powered fashion recommendation API built with FastAPI, CLIP, YOLOv8, Pinecone, and PostgreSQL.

## Quickstart

```bash
# 1. Copy and fill in env vars
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL (or use Docker)
docker compose up db -d

# 4. Seed the database
python -m db.seed

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
| `POST` | `/api/v1/recommend` | Upload outfit image, get recommendations |
| `POST` | `/api/v1/catalog/ingest` | Index a new product |
| `GET`  | `/api/v1/health` | Liveness + readiness check |

### Example: Get Recommendations

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
