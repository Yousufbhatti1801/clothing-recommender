# AI Clothing Recommender

An AI-powered fashion recommendation API. Upload any outfit photo and get visually similar products from a local catalog — filtered by budget and boosted for nearby sellers.

**Stack:** FastAPI · YOLOv8 (detection) · CLIP (embeddings) · Pinecone (vector search) · PostgreSQL (catalog) · Docker

---

## Quickstart

### Option A — Docker (recommended)

```bash
cp .env.example .env          # fill in PINECONE_API_KEY at minimum
docker compose up --build     # starts API + PostgreSQL together
```

API → http://localhost:8000  
Interactive docs → http://localhost:8000/docs

### Option B — Local development

```bash
# 1. Copy and configure environment
cp .env.example .env
#    Set PINECONE_API_KEY, adjust DB_URL if needed

# 2. Install runtime + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Start PostgreSQL
docker compose up db -d

# 4. Run database migrations
alembic upgrade head

# 5. Seed catalog (products + Pinecone vectors)
python scripts/seed_catalog.py
#    Flags: --reset  wipe & re-seed  |  --dry-run  validate image URLs only

# 6. Start the API
uvicorn app.main:app --reload
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `change-me-in-production` | Secret key for write endpoints (`X-API-Key` header) |
| `DEBUG` | `false` | Enable debug mode and verbose logs |
| `LOG_FORMAT` | `text` | `json` for cloud aggregators, `text` for local dev |
| `DB_URL` | `postgresql+asyncpg://…` | SQLAlchemy async database URL |
| `PINECONE_API_KEY` | — | **Required.** Pinecone API key |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Pinecone cloud region |
| `PINECONE_INDEX_NAME` | `clothing-embeddings` | Pinecone index name (created automatically) |
| `PINECONE_TOP_K` | `20` | Number of Pinecone nearest-neighbour results per query |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | HuggingFace CLIP model identifier |
| `CLIP_DEVICE` | `cpu` | `cpu` · `cuda` · `mps` |
| `YOLO_MODEL_PATH` | `ml/models/yolov8_fashion.pt` | Path to trained YOLOv8 weights |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.4` | Minimum detection confidence |
| `ALLOWED_IMAGE_HOSTS` | *(empty — all hosts)* | Comma-separated allowlist for catalog image downloads |
| `LOCALITY_BOOST` | `0.15` | Score bonus for sellers within `LOCALITY_RADIUS_KM` |
| `LOCALITY_RADIUS_KM` | `50.0` | Radius for local-seller preference |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON array of allowed CORS origins |
| `RATE_LIMIT_PER_MINUTE` | `20` | Max requests per IP per minute |
| `SENTRY_DSN` | *(empty)* | Optional Sentry DSN for error tracking |

---

## API Reference

All endpoints live under `/api/v1`.  Write operations require the `X-API-Key` header.

### Recommendation pipeline

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/pipeline/recommend` | — | Upload image → detect 6 categories → CLIP embed → Pinecone search |
| `POST` | `/recommend` | — | Upload image, returns DB-backed recommendations with locality boost |
| `POST` | `/detect` | — | YOLO-only detection, returns bounding boxes grouped by category |

**Pipeline recommend — form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `UploadFile` | ✅ | JPEG / PNG / WebP image (max 10 MB) |
| `budget` | `float` | — | Max price per item (USD); passed as Pinecone metadata filter |
| `user_latitude` | `float` | — | User latitude; enables local-seller boost |
| `user_longitude` | `float` | — | User longitude; enables local-seller boost |

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/recommend \
  -F "file=@outfit.jpg" \
  -F "budget=120" \
  -F "user_latitude=37.7749" \
  -F "user_longitude=-122.4194"
```

### Catalog management (requires `X-API-Key`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/catalog/ingest` | Embed and index one product |
| `POST` | `/catalog/ingest/batch` | Embed and index up to 100 products (partial success) |
| `GET` | `/catalog/products` | List products (`?category=shirt&limit=50&offset=0`) |
| `PATCH` | `/catalog/products/{id}` | Update product metadata (partial update) |
| `DELETE` | `/catalog/products/{id}` | Remove product from DB and Pinecone |

```bash
# Single ingest
curl -X POST http://localhost:8000/api/v1/catalog/ingest \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"Cool Jacket","brand":"Acme","category":"jacket","price":89.99,
       "currency":"USD","image_url":"https://example.com/jacket.jpg"}'

# Batch ingest
curl -X POST http://localhost:8000/api/v1/catalog/ingest/batch \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"products":[{"name":"Blue Dress",...},{"name":"Striped Skirt",...}]}'

# Partial update
curl -X PATCH http://localhost:8000/api/v1/catalog/products/<uuid> \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"price": 74.99}'
```

### Image upload

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload/image` | Validate and persist an image (returns a UUID filename) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + readiness check |

---

## Garment Categories

The pipeline detects and returns recommendations for all six categories:

| Category | Emoji | Pinecone namespace |
|----------|-------|-------------------|
| `shirt` | 👕 | `shirt` |
| `pants` | 👖 | `pants` |
| `shoes` | 👟 | `shoes` |
| `jacket` | 🧥 | `jacket` |
| `dress` | 👗 | `dress` |
| `skirt` | 🩱 | `skirt` |

---

## Testing

```bash
# Run all tests
pytest

# Skip slow integration tests
pytest -m "not integration" -v

# With coverage
pytest --cov=app --cov-report=html
```

---

## Linting

```bash
ruff check .
ruff format .
```

---

## Project Structure

```
clothing-recommender/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── api/routes/              # HTTP route handlers
│   │   ├── catalog.py           # Ingest, batch, list, update, delete
│   │   ├── detect.py            # YOLO detection endpoint
│   │   ├── pipeline.py          # Full AI recommendation pipeline
│   │   └── recommendation.py   # DB-backed recommendations
│   ├── services/                # Business logic
│   │   ├── detect_and_embed.py  # YOLO + CLIP pipeline step
│   │   ├── recommendation_pipeline.py  # Pinecone pipeline (6 categories, concurrent)
│   │   ├── catalog.py           # PostgreSQL CRUD
│   │   ├── ingestion.py         # Download → embed → index
│   │   └── vector_store.py      # Pinecone wrapper (upsert, query, delete)
│   ├── models/                  # ORM + Pydantic schemas
│   └── core/                    # Config, DB, Pinecone client, DI, logging
├── ml/
│   ├── clip_encoder.py          # CLIP singleton
│   ├── yolo_detector.py         # YOLOv8 singleton (6-category fashion model)
│   └── models/                  # Model weight files (git-ignored)
├── frontend/                    # Static SPA (served by FastAPI)
│   ├── index.html               # Budget + geo filters, 6-category result grids
│   ├── app.js                   # Fetch pipeline, render results
│   └── style.css
├── scripts/
│   └── seed_catalog.py          # Populate DB + Pinecone with sample products
├── tests/                       # Pytest suite (mocked ML + Pinecone)
├── alembic/                     # Database migrations
├── Dockerfile
├── docker-compose.yml
├── .env.example                 # Environment variable template
└── requirements.txt
```

---

## Troubleshooting

**Model weights missing (`FileNotFoundError: ml/models/yolov8_fashion.pt`)**  
The fashion-trained weights are not committed to git. Either train the model (`python scripts/train_fashion_yolo.py`) or place a compatible YOLOv8 checkpoint at `ml/models/yolov8_fashion.pt`. The base weights `yolov8n.pt` / `yolov8s.pt` in the repo root can be used as fallback.

**Pinecone cold start (first query is slow)**  
Pinecone serverless indexes may have a few seconds of cold-start latency on the first query after inactivity. Subsequent queries are fast.

**`PINECONE_API_KEY` not set**  
The app will start but vector searches will fail. Set the key in `.env` before running `seed_catalog.py` or making recommendation requests.

**Rate limit errors (`429 Too Many Requests`)**  
Default is 20 requests/minute/IP. Increase `RATE_LIMIT_PER_MINUTE` in `.env` for development.

**Docker image very large (multi-GB)**  
`torch` and `transformers` are large. The Dockerfile uses a multi-stage build — the final image includes only runtime dependencies. GPU support (`CLIP_DEVICE=cuda`) requires a CUDA-capable base image; adjust `Dockerfile` accordingly.


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
