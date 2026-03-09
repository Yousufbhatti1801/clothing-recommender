"""Geospatial helpers for locality-based ranking."""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two points
    using the Haversine formula.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def compute_locality_boost(
    user_lat: float,
    user_lon: float,
    seller_lat: float,
    seller_lon: float,
    radius_km: float = 50.0,
    boost: float = 0.15,
) -> float:
    """
    Return a score boost if the seller is within radius_km of the user,
    otherwise 0.0.

    The boost decays linearly from `boost` (at distance 0) to 0 (at radius_km).
    """
    distance = haversine_km(user_lat, user_lon, seller_lat, seller_lon)
    if distance >= radius_km:
        return 0.0
    return boost * (1.0 - distance / radius_km)
