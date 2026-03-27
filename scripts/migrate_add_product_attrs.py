#!/usr/bin/env python3
"""
scripts/migrate_add_product_attrs.py
=====================================
One-time schema migration: adds Phase 7 extended attribute columns to the
``products`` table and creates three performance indexes.

All statements use ``IF NOT EXISTS`` / ``IF NOT EXISTS`` so the script is
fully idempotent — safe to run multiple times.

Usage
-----
    python3.11 scripts/migrate_add_product_attrs.py

When to run
-----------
Run this ONCE on any existing database **before** starting the app with the
updated ORM (``app/models/orm.py``), or before running ``ingest_kaggle.py``.

Note: The FastAPI lifespan handler (``app/main.py``) also runs these
statements automatically at startup, so manual execution here is only
needed if you want to apply the migration without restarting the server
(e.g. in CI, staging, or admin scripts).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ordered list of idempotent SQL statements.
# Column additions must come before index creation.
MIGRATIONS: list[tuple[str, str]] = [
    # ── New attribute columns ──────────────────────────────────────────────
    ("ADD COLUMN colour",          "ALTER TABLE products ADD COLUMN IF NOT EXISTS colour VARCHAR(50)"),
    ("ADD COLUMN material",        "ALTER TABLE products ADD COLUMN IF NOT EXISTS material VARCHAR(100)"),
    ("ADD COLUMN style",           "ALTER TABLE products ADD COLUMN IF NOT EXISTS style VARCHAR(100)"),
    ("ADD COLUMN gender",          "ALTER TABLE products ADD COLUMN IF NOT EXISTS gender VARCHAR(20)"),
    ("ADD COLUMN size_available",  "ALTER TABLE products ADD COLUMN IF NOT EXISTS size_available TEXT"),
    ("ADD COLUMN occasion",        "ALTER TABLE products ADD COLUMN IF NOT EXISTS occasion VARCHAR(100)"),
    # ── Performance indexes ────────────────────────────────────────────────
    # ix_products_category  — used by every category-filtered query
    ("CREATE INDEX ix_products_category",       "CREATE INDEX IF NOT EXISTS ix_products_category ON products (category)"),
    # ix_products_category_price — budget filter + category search together
    ("CREATE INDEX ix_products_category_price", "CREATE INDEX IF NOT EXISTS ix_products_category_price ON products (category, price)"),
    # ix_products_colour — Phase 9 colour-attribute filtering
    ("CREATE INDEX ix_products_colour",         "CREATE INDEX IF NOT EXISTS ix_products_colour ON products (colour)"),
]


async def _run_migrations() -> None:
    from sqlalchemy import text

    from app.core.database import engine

    print("\n" + "─" * 56)
    print("  Products table — Phase 7 schema migration")
    print("─" * 56)

    async with engine.begin() as conn:
        for label, stmt in MIGRATIONS:
            print(f"  {label:<40} … ", end="", flush=True)
            await conn.execute(text(stmt))
            print("ok")

    print("─" * 56)
    print("  ✅  Migration complete — all statements applied.\n")


def main() -> None:
    t0 = time.perf_counter()
    asyncio.run(_run_migrations())
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s\n")


if __name__ == "__main__":
    main()
