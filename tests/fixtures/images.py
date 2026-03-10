"""
tests/fixtures/images.py
========================
Synthetic image factories for the integration test suite.

All functions return raw bytes in the requested format.  No real photography
assets are needed — PIL draws solid-colour rectangles which are perfectly
valid for testing image validation, preprocessing, and pipeline plumbing.
No GPU or ML model weights are required to generate these images.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


# ── JPEG ─────────────────────────────────────────────────────────────────────

def make_jpeg(width: int = 640, height: int = 480,
              color: tuple[int, int, int] = (110, 90, 70)) -> bytes:
    """Solid-colour JPEG (default 640×480, brownish)."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def make_outfit_jpeg() -> bytes:
    """640×960 JPEG with three colour bands simulating shirt / pants / shoes regions."""
    h, w = 960, 640
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 3, :]         = [70, 110, 200]   # shirt zone  (blue-ish)
    arr[h // 3: h * 2 // 3, :] = [30, 30,  80]  # pants zone  (dark indigo)
    arr[h * 2 // 3 :, :]     = [200, 180, 160]  # shoes zone  (light beige)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def make_oversized_jpeg() -> bytes:
    """2048×2048 JPEG — longer edge exceeds the 1024 px resize threshold."""
    return make_jpeg(2048, 2048, color=(64, 64, 64))


def make_tiny_jpeg() -> bytes:
    """50×50 JPEG — smaller than a garment bounding box; tests edge cases."""
    return make_jpeg(50, 50, color=(255, 255, 255))


# ── PNG / WebP ────────────────────────────────────────────────────────────────

def make_png(width: int = 320, height: int = 480) -> bytes:
    """Solid-colour PNG."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(180, 140, 100)).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def make_webp(width: int = 400, height: int = 600) -> bytes:
    """Solid-colour WebP — tests the third accepted MIME type."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(150, 180, 130)).save(buf, format="WEBP")
    buf.seek(0)
    return buf.getvalue()


# ── Invalid / corrupt ────────────────────────────────────────────────────────

def make_corrupt_jpeg() -> bytes:
    """Valid JPEG header but truncated payload — triggers decode failure."""
    good = make_jpeg(100, 100)
    return good[:50]  # cut off most of the file


def make_pdf_bytes() -> bytes:
    """Minimal fake PDF — wrong MIME type for the image endpoint."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"


def make_text_bytes() -> bytes:
    """Plain text bytes — wrong MIME type."""
    return b"This is not an image file.\n"


def make_empty_bytes() -> bytes:
    """Zero-length body — should be rejected by the image loader."""
    return b""
