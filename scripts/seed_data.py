#!/usr/bin/env python3
"""
MVP Data Seeder

A utility script to populate the local Clothing Recommender database and 
Pinecone index with sample products for testing the frontend MVP.

Usage:
    python scripts/seed_data.py
"""
import asyncio
import os
import sys

# Ensure app imports work by appending the root directory to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

# Load local environment variables to get the API key
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = "http://localhost:8000/api/v1/catalog/ingest"

if not API_KEY:
    print("❌ ERROR: API_KEY not found in environment. Please checking your .env file.")
    sys.exit(1)

# A curated list of sample high-quality Unsplash images representing clothing types.
# Note: In production, these should point to your own CDN/bucket.
SAMPLE_DATA = [
    # -- Shirts --
    {
        "name": "Classic White Cotton Tee",
        "brand": "Everlane",
        "description": "Essential crewneck, perfect for layering.",
        "category": "shirt",
        "price": 25.00,
        "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/white-tee",
    },
    {
        "name": "Vintage Denim Jacket",
        "brand": "Levi's",
        "description": "A timeless classic denim jacket for cool evenings.",
        "category": "shirt",
        "price": 89.99,
        "image_url": "https://images.unsplash.com/photo-1576871337622-98d48d1cf531?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/denim-jacket",
    },
    {
        "name": "Oversized Graphic Hoodie",
        "brand": "Champion",
        "description": "Cozy oversized hoodie with a retro graphic.",
        "category": "shirt",
        "price": 65.00,
        "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/graphic-hoodie",
    },
    
    # -- Pants --
    {
        "name": "Slim Fit Chinos",
        "brand": "J.Crew",
        "description": "Versatile khaki chinos for work or weekend wear.",
        "category": "pants",
        "price": 75.00,
        "image_url": "https://images.unsplash.com/photo-1624378439575-d1ead6bb293b?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/slim-chinos",
    },
    {
        "name": "Athletic Sweatpants",
        "brand": "Nike",
        "description": "Comfortable joggers for training and lounging.",
        "category": "pants",
        "price": 55.00,
        "image_url": "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/athletic-sweatpants",
    },
    {
        "name": "High-Waisted Skinny Jeans",
        "brand": "Madewell",
        "description": "Premium denim skinny jeans with a flattering high rise.",
        "category": "pants",
        "price": 128.00,
        "image_url": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/skinny-jeans",
    },

    # -- Shoes --
    {
        "name": "Classic Canvas Sneakers",
        "brand": "Converse",
        "description": "Iconic low-top canvas sneakers.",
        "category": "shoes",
        "price": 60.00,
        "image_url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/canvas-sneakers",
    },
    {
        "name": "Running Trainers",
        "brand": "Adidas",
        "description": "Lightweight mesh trainers for optimal performance.",
        "category": "shoes",
        "price": 130.00,
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/running-trainers",
    },
    {
        "name": "Leather Chelsea Boots",
        "brand": "Dr. Martens",
        "description": "Durable leather boots with a classic silhouette.",
        "category": "shoes",
        "price": 160.00,
        "image_url": "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?auto=format&fit=crop&q=80&w=600",
        "product_url": "https://example.com/products/chelsea-boots",
    },
]

async def seed_data():
    """Iterate through the sample data and ingest via the API."""
    print(f"🚀 Starting ingestion of {len(SAMPLE_DATA)} items...")
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        success_count = 0
        
        for item in SAMPLE_DATA:
            print(f"[{item['category'].upper()}] Ingesting: {item['name']}...", end=" ", flush=True)
            
            try:
                response = await client.post(API_URL, json=item, headers=headers)
                
                if response.status_code == 201:
                    print("✅ Success")
                    success_count += 1
                elif response.status_code == 429:
                    print("\n⏳ Rate limit hit! Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    # Retry once
                    response = await client.post(API_URL, json=item, headers=headers)
                    if response.status_code == 201:
                        print(f"  ↳ [{item['category'].upper()}] Retrieved from retry: {item['name']} ✅")
                        success_count += 1
                else:
                    print(f"❌ Failed ({response.status_code}): {response.text}")
                    
            except httpx.RequestError as e:
                print(f"❌ Connection error: {str(e)}")
            
            # Brief delay between requests to be polite to the backend image downloader
            await asyncio.sleep(1)
            
    print(f"\n🎉 Seeding complete! Successfully ingested {success_count}/{len(SAMPLE_DATA)} items.")


if __name__ == "__main__":
    try:
        asyncio.run(seed_data())
    except KeyboardInterrupt:
        print("\nSeeding interrupted by user.")
        sys.exit(0)
