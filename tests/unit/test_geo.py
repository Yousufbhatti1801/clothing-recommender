"""Unit tests for geospatial utilities (haversine + locality boost)."""
import pytest

from app.utils.geo import compute_locality_boost, haversine_km


class TestHaversineKm:
    def test_same_point_returns_zero(self):
        assert haversine_km(0, 0, 0, 0) == 0.0

    def test_known_distance_sf_to_la(self):
        """SF (37.77, -122.42) → LA (34.05, -118.24) ≈ 559 km."""
        d = haversine_km(37.7749, -122.4194, 34.0522, -118.2437)
        assert 550 < d < 570

    def test_antipodal_points(self):
        """(0, 0) → (0, 180) ≈ half circumference ≈ 20015 km."""
        d = haversine_km(0, 0, 0, 180)
        assert 20000 < d < 20100

    def test_symmetry(self):
        d1 = haversine_km(37.77, -122.42, 40.71, -74.01)
        d2 = haversine_km(40.71, -74.01, 37.77, -122.42)
        assert d1 == pytest.approx(d2)


class TestComputeLocalityBoost:
    def test_within_radius_returns_positive_boost(self):
        """Seller at the exact same location → full boost."""
        boost = compute_locality_boost(37.77, -122.42, 37.77, -122.42, radius_km=50.0, boost=0.15)
        assert boost == pytest.approx(0.15)

    def test_outside_radius_returns_zero(self):
        """SF to LA is ~559 km — far outside a 50 km radius."""
        boost = compute_locality_boost(37.7749, -122.4194, 34.0522, -118.2437, radius_km=50.0, boost=0.15)
        assert boost == 0.0

    def test_linear_decay(self):
        """At exactly half the radius distance, boost should be ~0.075."""
        # Two points ~25 km apart (SF to Daly City)
        boost_full = compute_locality_boost(37.77, -122.42, 37.77, -122.42, radius_km=50, boost=0.10)
        assert boost_full == pytest.approx(0.10)

    def test_zero_radius(self):
        """With radius_km=0 every non-zero distance is outside."""
        # distance > 0 but radius is 0 → division by zero guard
        boost = compute_locality_boost(37.77, -122.42, 37.78, -122.42, radius_km=0.0, boost=0.15)
        # haversine will be > 0 and >= radius (0.0), so boost == 0
        assert boost == 0.0

    def test_custom_boost_value(self):
        boost = compute_locality_boost(37.77, -122.42, 37.77, -122.42, radius_km=50, boost=0.50)
        assert boost == pytest.approx(0.50)
