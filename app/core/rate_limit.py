"""Shared rate limiter configuration."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared across all routes and registered in main.py
limiter = Limiter(key_func=get_remote_address)
