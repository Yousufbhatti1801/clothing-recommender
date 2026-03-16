# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps for Pillow + OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy pre-built dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application source (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Pre-download CLIP weights at build time (speeds up cold start)
# The || true ensures a build failure here doesn't stop the entire build
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" || true

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
