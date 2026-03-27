# 🚀 AI Fashion Recommender — Engineering Improvement Plan

> **Author**: Senior AI Engineer Assessment  
> **Date**: Session 2 — Post-MVP  
> **Baseline**: 274 tests passing · 37 catalog items · CLIP ViT-B/32 · YOLOv8-nano  
> **Last test output** (multi-region fallback):  
> - Pants 77% · Shoes 70% · Jacket 74% top matches  
> - 3 detections, 12 total matches from a single outfit image

---

## Executive Diagnosis

The system has a working end-to-end pipeline (upload → detect → embed → search → rank), but recommendation accuracy is **fundamentally limited** by three structural problems:

| Root Cause | Impact | Severity |
|---|---|---|
| **37 products** across 6 categories (3–8 per namespace) | Vector search returns the *only* items that exist, regardless of similarity | 🔴 Critical |
| **CLIP ViT-B/32** — smallest CLIP variant (63M params) | Embeddings lack fine-grained fashion discrimination (colour, texture, cut) | 🔴 Critical |
| **No fashion metadata** — no colour, material, style, or size | Cannot filter or re-rank by attributes users actually care about | 🟠 High |

Fixing these three alone would produce a **step-change** in perceived accuracy. Everything else is refinement.

---

## Phase 7 — Catalog Expansion (1,000+ Products)

> **Goal**: Make vector search meaningful. With 8 shoes, *every* shoe gets returned regardless of similarity.

### 7.1 Build a real catalog scraper

```
scripts/
  scrape_catalog.py      # NEW — async scraper with rate limiting
  catalog_sources.json   # NEW — URLs/configs for data sources
```

**Data sources** (free, no API key):

| Source | What | Volume |
|---|---|---|
| Kaggle "Fashion Product Images" dataset | 44K products with images + metadata | ⭐ Best starting point |
| H&M Personalized Fashion (Kaggle) | 105K articles, images, attributes | Large, rich metadata |
| DeepFashion (academic) | 800K images, bbox + attributes | Academic, requires registration |
| Unsplash API (existing approach) | High-quality photos, limited metadata | Small but clean |

**Implementation steps**:
1. Download Kaggle "Fashion Product Images Small" (280MB, ~44K items)
2. Build ETL script: read CSV → map categories → download images → CLIP encode → upsert to Pinecone + PostgreSQL
3. Add proper product metadata: `colour`, `material`, `style`, `gender`, `season`
4. Target: **1,000 products minimum** (200 per core category) for Phase 7

### 7.2 Enrich the Product schema

**File**: `app/models/orm.py`

```python
# ADD these columns to Product
colour: Mapped[str | None] = mapped_column(String(50))     # "navy", "white", "red"
material: Mapped[str | None] = mapped_column(String(100))   # "cotton", "leather", "denim"
style: Mapped[str | None] = mapped_column(String(100))      # "casual", "formal", "streetwear"
gender: Mapped[str | None] = mapped_column(String(20))       # "men", "women", "unisex"
size_available: Mapped[str | None] = mapped_column(Text)     # JSON: ["S","M","L","XL"]
occasion: Mapped[str | None] = mapped_column(String(100))    # "office", "casual", "party"
```

### 7.3 Add database indexes

**File**: `app/models/orm.py`

```python
from sqlalchemy import Index
# On Product table:
__table_args__ = (
    Index("ix_products_category", "category"),
    Index("ix_products_category_price", "category", "price"),
    Index("ix_products_colour", "colour"),
)
```

### 7.4 Store metadata in Pinecone vectors

Currently upserting: `{price, brand, image_url, seller_id}`.  
**Add**: `colour`, `material`, `style`, `gender` → enables server-side filtering.

**Expected impact**: Accuracy jumps from ~70% top-1 to meaningful results because the search space isn't exhausted.

---

## Phase 8 — Upgrade CLIP Model

> **Goal**: Better embedding quality = better similarity = better recommendations.

### 8.1 Model comparison

| Model | Params | Dim | ImageNet Zero-shot | Fashion Discriminability | Inference (CPU) |
|---|---|---|---|---|---|
| **ViT-B/32** (current) | 63M | 512 | 63.2% | Low | ~100ms |
| **ViT-B/16** | 86M | 512 | 68.3% | Medium | ~200ms |
| **ViT-L/14** | 304M | 768 | 75.5% | High | ~800ms |
| **SigLIP ViT-B/16** | 86M | 768 | 73.4% | High (better at details) | ~200ms |
| **FashionCLIP** (patrickjohncyh) | 86M | 512 | N/A | ⭐ Best for fashion | ~200ms |

**Recommended path**: 
1. **Immediate**: Swap to **`patrickjohncyh/fashion-clip`**  — fine-tuned on fashion data, same 512-d, minimal code change
2. **Later**: Upgrade to **ViT-L/14** (768-d) for maximum quality when GPU is available

### 8.2 Implementation

**File**: `app/core/config.py`
```python
clip_model_name: str = "patrickjohncyh/fashion-clip"  # was openai/clip-vit-base-patch32
```

**File**: `ml/clip_encoder.py` — No code change needed (HuggingFace auto-detects architecture)

⚠️ **Critical**: When changing the model, **re-embed the entire catalog**:
```bash
python3 scripts/reindex_catalog.py --model patrickjohncyh/fashion-clip
```

### 8.3 Dimension change plan (for ViT-L/14 later)

If upgrading to 768-d later:
1. Create new Pinecone index `clothing-embeddings-v2` with `dimension=768`
2. Re-embed all products
3. Blue-green switch via config: `pinecone_index_name`
4. Delete old index

**Expected impact**: FashionCLIP alone gives ~15-20% improvement in fashion-specific retrieval. ViT-L/14 adds another ~12%.

---

## Phase 9 — Colour & Attribute Extraction

> **Goal**: Users care about "find me a *blue* jacket similar to this" — pure embeddings lose colour info.

### 9.1 CLIP-based colour extraction

**New file**: `ml/attribute_extractor.py`

```python
COLOUR_PROMPTS = [
    "a black garment", "a white garment", "a red garment",
    "a blue garment", "a navy garment", "a green garment",
    "a brown garment", "a grey garment", "a beige garment",
    "a pink garment", "a yellow garment", "a orange garment",
]
# Zero-shot classify cropped garment → dominant colour
```

### 9.2 Hybrid search: vector + attribute filter

**File**: `app/services/recommendation_pipeline.py`

```python
# Pinecone filter becomes:
filter = {
    "$and": [
        {"price": {"$lte": budget}},
        {"colour": {"$in": [detected_colour, "multi"]}},  # soft match
    ]
}
```

### 9.3 Material/style extraction (bonus)

Use CLIP zero-shot with prompts:
- Material: `"a cotton garment"`, `"a leather garment"`, `"a denim garment"`, ...
- Style: `"a casual outfit"`, `"a formal outfit"`, `"streetwear"`, ...

Store as Pinecone metadata → enable API query params: `?colour=blue&style=casual`

**Expected impact**: Eliminates the "I uploaded a white sneaker, got a black oxford" problem. ~20% accuracy improvement for colour-sensitive queries.

---

## Phase 10 — Re-ranking & Result Diversity

> **Goal**: Raw cosine similarity isn't the best ranking signal. Add diversity and cross-category coherence.

### 10.1 MMR (Maximal Marginal Relevance)

Currently: return top-5 by cosine score.  
Problem: all 5 results can be nearly identical (e.g., 5 slim blue jeans).

**New file**: `app/services/reranker.py`

```python
def mmr_rerank(query_vec, candidate_vecs, scores, lambda_=0.7, top_k=5):
    """
    Iteratively select candidates that are:
      - Similar to the query (relevance)
      - Dissimilar to already-selected items (diversity)
    """
    # Standard MMR: score = λ * sim(q, d) - (1-λ) * max(sim(d, d_selected))
```

### 10.2 Cross-category outfit coherence

Currently: shirt, pants, shoes are recommended **independently**.  
Better: if user uploads a "smart casual" outfit, recommend items that go *together*.

**Approach**:
1. Compute "outfit style vector" = mean of all detected crop embeddings
2. For each category's top-10 Pinecone results, re-rank by `0.7 * category_sim + 0.3 * outfit_sim`
3. This naturally pushes casual pants higher when the shirt is casual

### 10.3 Fix locality boost

**Current** (broken): `final_score = cosine_sim + 0.15` → additive, can push unrelated local items above good remote ones.

**Fix**: multiplicative boost
```python
# In app/services/recommendation.py
final_score = match.score * (1.0 + locality_boost)  # max 15% uplift
```

**Expected impact**: Users see varied, coherent results instead of 5 variations of the same item.

---

## Phase 11 — Detection Quality Improvements

> **Goal**: Detect garments more reliably, especially from non-standard images.

### 11.1 Upgrade YOLO model

| Model | Size | mAP (fashion) | Speed (CPU) |
|---|---|---|---|
| **yolov8n** (current) | 3.2M | ~40% | ~150ms |
| **yolov8s** | 11.2M | ~55% | ~300ms |
| **yolov8m** | 25.9M | ~62% | ~700ms |
| YOLOv8 fine-tuned on DeepFashion2 | ~25M | ~75% | ~700ms |

**Recommendation**: Upgrade to **yolov8s** (`keremberke/yolov8s-fashion-detection`) immediately — 2× better accuracy, still fast on CPU.

### 11.2 Adaptive body region splitting

Current rigid splits fail on:
- Cropped photos (only upper body)
- Wide-angle shots (person is small)
- Seated poses

**Fix**: Use person detection first (MediaPipe Pose or YOLO-pose), then derive body zones from keypoints:
```python
# If hip keypoint at 60% of image height → adjust pants zone to 50-85%
# If shoulders at 10% → person occupies bottom 90% → shift all zones
```

### 11.3 Multi-model ensemble

Run YOLO + CLIP classification in parallel, merge results:
```python
yolo_detections = await yolo.detect(image)
clip_regions = await clip_classifier.classify_regions(image)

# Union: keep YOLO detections where confident, fill gaps with CLIP
merged = merge_detections(yolo_detections, clip_regions, iou_threshold=0.3)
```

### 11.4 Confidence calibration

YOLO confidence and CLIP zero-shot scores are on **different scales** (YOLO: 0.4-0.95, CLIP: 0.18-0.35). Normalize both to a common [0,1] scale:
```python
# Temperature-scaled sigmoid for CLIP scores
calibrated = 1 / (1 + exp(-10 * (raw_score - 0.22)))
```

**Expected impact**: Detect 4-5 garments per outfit image instead of 1-3. Fewer false negatives.

---

## Phase 12 — Caching & Performance

> **Goal**: Sub-second responses for repeated/similar queries.

### 12.1 Redis embedding cache

**New file**: `app/core/redis.py`

Cache CLIP embeddings keyed by image content hash:
```python
cache_key = f"clip:{sha256(image_bytes)}"
cached = redis.get(cache_key)
if cached:
    return np.frombuffer(cached, dtype=np.float32)
# else: encode, cache for 24h, return
```

### 12.2 Pinecone result cache

Cache Pinecone query results (namespace + vector hash → results):
- TTL: 1 hour (catalog changes are infrequent)
- Invalidate on catalog update

### 12.3 Upload cleanup

Currently `uploads/` grows unbounded.

**Fix**: Background task that deletes files older than 1 hour:
```python
@app.on_event("startup")
@repeat_every(seconds=3600)
async def cleanup_uploads():
    cutoff = time.time() - 3600
    for f in Path("uploads").iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink()
```

### 12.4 ML executor tuning

Current: `ThreadPoolExecutor(max_workers=2)`.  
With Redis cache reducing actual CLIP calls by ~60%, increase to `max_workers=4` for parallel YOLO + CLIP.

**Expected impact**: P95 latency from ~2s to ~400ms for cached queries. Unbounded disk growth eliminated.

---

## Phase 13 — User Experience & Personalization

> **Goal**: Remember user preferences, enable text search, add filtering UI support.

### 13.1 User accounts & preference learning

**New ORM models**:
```python
class User(Base):           # id, email, location, preferences_json
class SearchHistory(Base):  # user_id, query_embedding, results, timestamp
class Feedback(Base):       # user_id, product_id, action (click/save/dismiss), timestamp
```

**Personalization**: After 10+ interactions, compute "user style vector" = weighted average of clicked product embeddings → blend with query: `0.8 * query + 0.2 * user_style`

### 13.2 Text query support

Users should be able to type "blue denim jacket under $100":

**File**: `app/api/routes/recommendations.py`
```python
@router.post("/search/text")
async def text_search(query: str, budget: float = None):
    text_embedding = clip_encoder.encode_text(query)
    # Search Pinecone with text embedding + price filter
```

CLIP already supports text→image similarity out of the box.

### 13.3 API filtering parameters

Add query params to the pipeline endpoint:
```
POST /api/v1/recommend/pipeline?colour=blue&gender=men&style=casual&budget=100
```

These map directly to Pinecone metadata filters added in Phase 9.

**Expected impact**: Moves from "dumb image match" to "intelligent fashion assistant".

---

## Phase 14 — Production Hardening

> **Goal**: Production-ready deployment with monitoring, security, and scalability.

### 14.1 GPU inference
- Dockerfile with CUDA support
- Config: `clip_device: "cuda"` → 10× faster embedding
- Batch inference: accumulate requests for 50ms, batch-encode

### 14.2 Observability
- Structured JSON logging (already configurable, not enabled)
- Sentry integration (DSN field exists, not configured)
- Prometheus metrics: `/metrics` endpoint
  - `recommendation_latency_seconds` histogram
  - `detection_count` counter by method (yolo/clip_fallback)
  - `pinecone_query_latency_seconds` histogram

### 14.3 Security
- Rate limiting per API key (currently global 20/min)
- Image upload validation (max size, allowed formats, virus scan)
- Input sanitization for text queries
- CORS tightening for production

### 14.4 Price as Numeric
**File**: `app/models/orm.py`
```python
from sqlalchemy import Numeric
price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
```
Avoids floating-point rounding on $99.99 → $99.98999...

### 14.5 Consolidate dual recommendation endpoints

Currently two overlapping endpoints:
- `POST /recommend` → `RecommendationService` (PostgreSQL-backed)
- `POST /recommend/pipeline` → `RecommendationPipeline` (Pinecone-only)

**Merge into one unified pipeline** that uses Pinecone for search + PostgreSQL for enrichment.

---

## Priority Execution Order

```
Phase  7: Catalog Expansion        ██████████  ← DO THIS FIRST (biggest accuracy gain)
Phase  8: Upgrade CLIP Model       ████████    ← Immediate win, 1 config change + re-index
Phase  9: Colour/Attribute Extract  ██████     ← Eliminates wrong-colour recommendations
Phase 10: Re-ranking & Diversity   █████      ← Makes results look "smart"
Phase 11: Detection Improvements   █████      ← More garments detected per image
Phase 12: Caching & Performance    ████       ← Latency & cost reduction
Phase 13: Personalization          ████       ← User retention & engagement
Phase 14: Production Hardening     ███        ← Before any public launch
```

---

## Estimated Impact by Phase

| Phase | Accuracy Δ | Latency Δ | Effort |
|---|---|---|---|
| 7 — Catalog (1K+ products) | **+30-40%** | — | 2-3 days |
| 8 — FashionCLIP | **+15-20%** | +50ms | 2 hours |
| 9 — Colour extraction | **+15-20%** | +30ms | 1-2 days |
| 10 — MMR + outfit coherence | **+10-15%** | +10ms | 1 day |
| 11 — Better detection | **+10%** | +100ms | 2 days |
| 12 — Redis cache | — | **-60%** | 1 day |
| 13 — Personalization | **+10%** (returning users) | — | 3-4 days |
| 14 — Production hardening | — | — | 2-3 days |

**Total estimated accuracy improvement**: From ~70% top-1 relevance → **~90%+ top-5 relevance** after Phases 7-11.

---

## Quick Wins (Can Do Today)

1. **Swap to FashionCLIP** — 1 line config change + re-embed (Phase 8)
2. **Upgrade YOLO to yolov8s** — 1 line config change (Phase 11.1)
3. **Fix locality boost to multiplicative** — 1 line change (Phase 10.3)
4. **Add DB indexes** — 3 lines in ORM (Phase 7.3)
5. **Download Kaggle dataset** — immediate catalog expansion (Phase 7.1)

---

## Architecture After All Phases

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Gateway                      │
│  /recommend  /search/text  /catalog  /health            │
└──────┬────────────┬───────────────┬──────────────────┘
       │            │               │
 ┌─────▼─────┐ ┌───▼─────┐  ┌─────▼──────┐
 │  YOLO v8s │ │ CLIP    │  │ Attribute  │
 │  Detector │ │ Fashion │  │ Extractor  │
 │  +Pose    │ │ Encoder │  │ (colour,   │
 │  fallback │ │ 512-d   │  │  material) │
 └─────┬─────┘ └───┬─────┘  └─────┬──────┘
       │            │               │
       └────────────┼───────────────┘
                    │
            ┌───────▼───────┐
            │  Redis Cache  │
            │  (embeddings) │
            └───────┬───────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
 ┌─────▼─────┐ ┌───▼──────┐ ┌──▼──────────┐
 │ Pinecone  │ │PostgreSQL│ │  Re-ranker   │
 │ Vector DB │ │ Catalog  │ │  MMR + Outfit│
 │ +metadata │ │ +Users   │ │  Coherence   │
 │  filters  │ │ +History │ │              │
 └───────────┘ └──────────┘ └──────────────┘
```

---

*This plan is sequenced so that each phase builds on the previous one. Phase 7 (catalog) is the single highest-impact change — everything else improves the quality of results from a pool that's already large enough to contain genuinely similar items.*
