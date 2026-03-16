"""
tests/test_yolo_model_loading.py
=================================
Validate that the YOLO fashion model loads correctly, its class names
match the app schema, and all dynamic wiring (label map, is_fashion_model,
model_summary) works as expected.

These tests use the actual model file on disk — they are NOT mocked.
Mark them slow if real-model tests should be deselected in CI.
"""
from __future__ import annotations

import pytest
from PIL import Image

from app.models.schemas import GarmentCategory
from ml.fashion_classes import (
    APP_CLASS_NAMES,
    build_label_map_from_model_names,
    is_fashion_model,
    name_to_category,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def yolo_model():
    """Load the YOLO model once for the whole module."""
    from ultralytics import YOLO
    return YOLO("ml/models/yolov8_fashion.pt")


@pytest.fixture(scope="module")
def yolo_detector():
    """Load the YOLODetector singleton for the whole module."""
    import os
    os.environ.setdefault("PINECONE_API_KEY", "dummy")
    import warnings
    warnings.filterwarnings("ignore")
    from ml.yolo_detector import YOLODetector
    return YOLODetector()


# ══════════════════════════════════════════════════════════════════════════════
# Test class: Model file loading
# ══════════════════════════════════════════════════════════════════════════════

class TestModelLoading:
    """Verify the raw YOLO model file can be loaded and inspected."""

    def test_model_loads_successfully(self, yolo_model):
        assert yolo_model is not None

    def test_model_has_names_dict(self, yolo_model):
        assert isinstance(yolo_model.names, dict)
        assert len(yolo_model.names) > 0

    def test_model_has_13_classes(self, yolo_model):
        assert len(yolo_model.names) == 13, (
            f"Expected 13 fashion classes, got {len(yolo_model.names)}: {yolo_model.names}"
        )

    def test_class_names_match_app_schema(self, yolo_model):
        """Every class name in the model must match APP_CLASS_NAMES exactly."""
        for idx, name in yolo_model.names.items():
            assert name == APP_CLASS_NAMES[idx], (
                f"Class {idx}: model has '{name}', expected '{APP_CLASS_NAMES[idx]}'"
            )

    def test_class_names_order_matches_yaml(self, yolo_model):
        """The order of names matches config/fashion_dataset.yaml."""
        expected_order = [
            "shirt", "t-shirt", "pants", "jeans", "shorts",
            "shoes", "sneakers", "jacket", "coat", "dress",
            "skirt", "bag", "hat",
        ]
        actual_order = [yolo_model.names[i] for i in range(13)]
        assert actual_order == expected_order


# ══════════════════════════════════════════════════════════════════════════════
# Test class: Dynamic label map
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicLabelMap:
    """Verify build_label_map_from_model_names resolves correctly."""

    def test_label_map_has_all_indices(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        for idx in range(13):
            assert idx in label_map, f"Missing class index {idx} in label map"

    def test_shirt_indices_map_to_shirt(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        # 0: shirt, 1: t-shirt → both SHIRT
        assert label_map[0] == GarmentCategory.SHIRT
        assert label_map[1] == GarmentCategory.SHIRT

    def test_pants_indices_map_to_pants(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        # 2: pants, 3: jeans, 4: shorts → all PANTS
        assert label_map[2] == GarmentCategory.PANTS
        assert label_map[3] == GarmentCategory.PANTS
        assert label_map[4] == GarmentCategory.PANTS

    def test_shoes_indices_map_to_shoes(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        # 5: shoes, 6: sneakers → both SHOES
        assert label_map[5] == GarmentCategory.SHOES
        assert label_map[6] == GarmentCategory.SHOES

    def test_jacket_indices_map_to_jacket(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        # 7: jacket, 8: coat → both JACKET
        assert label_map[7] == GarmentCategory.JACKET
        assert label_map[8] == GarmentCategory.JACKET

    def test_dress_maps_to_dress(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        assert label_map[9] == GarmentCategory.DRESS

    def test_skirt_maps_to_skirt(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        assert label_map[10] == GarmentCategory.SKIRT

    def test_accessories_map_to_other(self, yolo_model):
        label_map = build_label_map_from_model_names(yolo_model.names)
        # 11: bag, 12: hat → both OTHER
        assert label_map[11] == GarmentCategory.OTHER
        assert label_map[12] == GarmentCategory.OTHER

    def test_no_coco_indices_remain(self, yolo_model):
        """The label map should NOT contain COCO index → fashion category
        coincidences (the old bug: COCO 0='person' → SHIRT)."""
        label_map = build_label_map_from_model_names(yolo_model.names)
        # In the new model, index 0 is 'shirt' — that's correct.
        # But we verify it's because the NAME is 'shirt', not because
        # the index happens to be 0.
        for idx, cat in label_map.items():
            name = yolo_model.names[idx]
            resolved = name_to_category(name)
            assert cat == resolved, (
                f"Index {idx}: label map says {cat} but name_to_category('{name}') = {resolved}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Test class: is_fashion_model
# ══════════════════════════════════════════════════════════════════════════════

class TestIsFashionModel:
    """Verify the fashion model detection flag."""

    def test_fashion_model_detected(self, yolo_model):
        assert is_fashion_model(yolo_model.names) is True

    def test_coco_model_not_fashion(self):
        coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
        }
        assert is_fashion_model(coco_names) is False

    def test_partial_fashion_names_detected(self):
        partial_names = {0: "shirt", 1: "car", 2: "airplane"}
        assert is_fashion_model(partial_names) is True


# ══════════════════════════════════════════════════════════════════════════════
# Test class: YOLODetector integration
# ══════════════════════════════════════════════════════════════════════════════

class TestYOLODetectorIntegration:
    """Verify the YOLODetector class wiring with the real model."""

    def test_detector_loads(self, yolo_detector):
        assert yolo_detector is not None

    def test_detector_is_fashion_model(self, yolo_detector):
        assert yolo_detector.is_fashion_model is True

    def test_detector_has_label_map(self, yolo_detector):
        assert len(yolo_detector._label_map) == 13

    def test_model_summary_structure(self, yolo_detector):
        summary = yolo_detector.model_summary()
        assert "model_path" in summary
        assert "is_fashion_model" in summary
        assert "num_classes" in summary
        assert "class_names" in summary
        assert "confidence_threshold" in summary
        assert summary["is_fashion_model"] is True
        assert summary["num_classes"] == 13

    def test_model_summary_class_names(self, yolo_detector):
        summary = yolo_detector.model_summary()
        class_names = summary["class_names"]
        assert len(class_names) == 13
        assert class_names[0] == "shirt"

    def test_detect_returns_list(self, yolo_detector):
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        result = yolo_detector.detect(img)
        assert isinstance(result, list)

    def test_detect_targets_returns_list(self, yolo_detector):
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        result = yolo_detector.detect_targets(img)
        assert isinstance(result, list)

    def test_detect_all_fashion_returns_list(self, yolo_detector):
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        result = yolo_detector.detect_all_fashion(img)
        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════════════════════
# Test class: name_to_category resolver
# ══════════════════════════════════════════════════════════════════════════════

class TestNameToCategory:
    """Verify the name→category resolver for various input strings."""

    @pytest.mark.parametrize("name,expected", [
        ("shirt",               GarmentCategory.SHIRT),
        ("t-shirt",             GarmentCategory.SHIRT),
        ("pants",               GarmentCategory.PANTS),
        ("jeans",               GarmentCategory.PANTS),
        ("shorts",              GarmentCategory.PANTS),
        ("shoes",               GarmentCategory.SHOES),
        ("sneakers",            GarmentCategory.SHOES),
        ("jacket",              GarmentCategory.JACKET),
        ("coat",                GarmentCategory.JACKET),
        ("dress",               GarmentCategory.DRESS),
        ("skirt",               GarmentCategory.SKIRT),
        ("bag",                 GarmentCategory.OTHER),
        ("hat",                 GarmentCategory.OTHER),
    ])
    def test_app_class_names(self, name, expected):
        assert name_to_category(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("short sleeve top",    GarmentCategory.SHIRT),
        ("long sleeve top",     GarmentCategory.SHIRT),
        ("trousers",            GarmentCategory.PANTS),
        ("short sleeve dress",  GarmentCategory.DRESS),
    ])
    def test_deepfashion2_names(self, name, expected):
        assert name_to_category(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("denim jeans",         GarmentCategory.PANTS),
        ("oxford shoe",         GarmentCategory.SHOES),
        ("polo shirt",          GarmentCategory.SHIRT),
        ("running sneaker",     GarmentCategory.SHOES),
    ])
    def test_keyword_fallback(self, name, expected):
        assert name_to_category(name) == expected

    def test_unknown_returns_other(self):
        assert name_to_category("banana") == GarmentCategory.OTHER
        assert name_to_category("helicopter") == GarmentCategory.OTHER
