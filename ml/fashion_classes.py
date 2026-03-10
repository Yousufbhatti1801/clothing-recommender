"""
ml/fashion_classes.py
======================
Centralised class definitions for clothing detection across multiple datasets.

This module is the single source of truth for how raw class names (strings
from any YOLO model's ``model.names`` dict) are translated to the application's
``GarmentCategory`` enum.

Supported source schemas
------------------------
APP_NATIVE      — 13-class schema used when training ``yolov8_fashion.pt``.
                  Class order MUST match ``config/fashion_dataset.yaml``.
DEEPFASHION2    — 13 categories from the DeepFashion2 benchmark dataset.
                  https://github.com/switchablenorms/DeepFashion2
FASHIONPEDIA    — 27 super-categories from the Fashionpedia benchmark.
                  https://fashionpedia.github.io/home/
COCO            — COCO-80 accessory classes (handbag, tie, backpack).

Usage
-----
    from ml.fashion_classes import build_label_map_from_model_names
    label_map = build_label_map_from_model_names(model.names)
    # → {0: GarmentCategory.SHIRT, 1: GarmentCategory.PANTS, ...}
"""
from __future__ import annotations

from app.models.schemas import GarmentCategory

# ══════════════════════════════════════════════════════════════════════════════
# App-native class names
# These 13 names are used when training yolov8_fashion.pt from scratch or
# via fine-tuning.  The ORDER must match config/fashion_dataset.yaml nc entries.
# ══════════════════════════════════════════════════════════════════════════════

APP_CLASS_NAMES: list[str] = [
    "shirt",      # 0  — short/long sleeve tops
    "t-shirt",    # 1  — crew-neck tees
    "pants",      # 2  — generic trousers
    "jeans",      # 3  — denim trousers
    "shorts",     # 4  — short trousers
    "shoes",      # 5  — generic footwear
    "sneakers",   # 6  — trainers / athletic shoes
    "jacket",     # 7  — blazers / windbreakers
    "coat",       # 8  — overcoats / heavy outerwear
    "dress",      # 9  — full-length dresses
    "skirt",      # 10 — skirts
    "bag",        # 11 — handbags / backpacks
    "hat",        # 12 — caps / hats
]

# ── App-native name → GarmentCategory ────────────────────────────────────────

APP_NAME_TO_CATEGORY: dict[str, GarmentCategory] = {
    "shirt":    GarmentCategory.SHIRT,
    "t-shirt":  GarmentCategory.SHIRT,
    "pants":    GarmentCategory.PANTS,
    "jeans":    GarmentCategory.PANTS,
    "shorts":   GarmentCategory.PANTS,
    "shoes":    GarmentCategory.SHOES,
    "sneakers": GarmentCategory.SHOES,
    "jacket":   GarmentCategory.JACKET,
    "coat":     GarmentCategory.JACKET,
    "dress":    GarmentCategory.DRESS,
    "skirt":    GarmentCategory.SKIRT,
    "bag":      GarmentCategory.OTHER,
    "hat":      GarmentCategory.OTHER,
}

# ══════════════════════════════════════════════════════════════════════════════
# DeepFashion2 class mapping
# Source: https://github.com/switchablenorms/DeepFashion2
# 13 classes, category_id 1–13 in the JSON (converted to 0–12 for YOLO).
# ══════════════════════════════════════════════════════════════════════════════

# Maps DeepFashion2 category name → (YOLO 0-indexed class id, GarmentCategory)
DEEPFASHION2_CLASSES: dict[int, tuple[str, GarmentCategory]] = {
    0:  ("short sleeve top",     GarmentCategory.SHIRT),
    1:  ("long sleeve top",      GarmentCategory.SHIRT),
    2:  ("short sleeve outwear", GarmentCategory.JACKET),
    3:  ("long sleeve outwear",  GarmentCategory.JACKET),
    4:  ("vest",                 GarmentCategory.SHIRT),
    5:  ("sling",                GarmentCategory.SHIRT),
    6:  ("shorts",               GarmentCategory.PANTS),
    7:  ("trousers",             GarmentCategory.PANTS),
    8:  ("skirt",                GarmentCategory.SKIRT),
    9:  ("short sleeve dress",   GarmentCategory.DRESS),
    10: ("long sleeve dress",    GarmentCategory.DRESS),
    11: ("vest dress",           GarmentCategory.DRESS),
    12: ("sling dress",          GarmentCategory.DRESS),
}

DEEPFASHION2_NAME_TO_CATEGORY: dict[str, GarmentCategory] = {
    name: cat for _, (name, cat) in DEEPFASHION2_CLASSES.items()
}

# ══════════════════════════════════════════════════════════════════════════════
# Fashionpedia class mapping
# Source: https://fashionpedia.github.io/home/
# 27 super-categories, used in YOLOS-Fashionpedia
# (valentinafeve/yolos-fashionpedia on HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════

FASHIONPEDIA_CLASSES: list[tuple[str, GarmentCategory]] = [
    ("shirt, blouse",                GarmentCategory.SHIRT),
    ("top, t-shirt, sweatshirt",     GarmentCategory.SHIRT),
    ("sweater",                      GarmentCategory.SHIRT),
    ("cardigan",                     GarmentCategory.JACKET),
    ("jacket",                       GarmentCategory.JACKET),
    ("vest",                         GarmentCategory.SHIRT),
    ("pants",                        GarmentCategory.PANTS),
    ("shorts",                       GarmentCategory.PANTS),
    ("skirt",                        GarmentCategory.SKIRT),
    ("coat",                         GarmentCategory.JACKET),
    ("dress",                        GarmentCategory.DRESS),
    ("jumpsuit",                     GarmentCategory.DRESS),
    ("cape",                         GarmentCategory.JACKET),
    ("glasses",                      GarmentCategory.OTHER),
    ("hat",                          GarmentCategory.OTHER),
    ("headband, hair bow",           GarmentCategory.OTHER),
    ("hair accessory",               GarmentCategory.OTHER),
    ("tie",                          GarmentCategory.OTHER),
    ("glove",                        GarmentCategory.OTHER),
    ("watch",                        GarmentCategory.OTHER),
    ("belt",                         GarmentCategory.OTHER),
    ("leg warmer",                   GarmentCategory.OTHER),
    ("tights, stockings",            GarmentCategory.OTHER),
    ("sock",                         GarmentCategory.OTHER),
    ("shoe, boot, sandal, slipper",  GarmentCategory.SHOES),
    ("bag, wallet",                  GarmentCategory.OTHER),
    ("scarf",                        GarmentCategory.OTHER),
]

FASHIONPEDIA_NAME_TO_CATEGORY: dict[str, GarmentCategory] = {
    name: cat for name, cat in FASHIONPEDIA_CLASSES
}

# ══════════════════════════════════════════════════════════════════════════════
# COCO-80 partial mapping (accessory classes only)
# Used as fallback when a COCO-trained model is loaded.
# Most COCO classes return OTHER; a handful are fashion-adjacent.
# ══════════════════════════════════════════════════════════════════════════════

COCO_NAME_TO_CATEGORY: dict[str, GarmentCategory] = {
    "backpack":  GarmentCategory.OTHER,   # 24
    "umbrella":  GarmentCategory.OTHER,   # 25
    "handbag":   GarmentCategory.OTHER,   # 26
    "tie":       GarmentCategory.OTHER,   # 27
    "suitcase":  GarmentCategory.OTHER,   # 28
    "skis":      GarmentCategory.OTHER,   # 30
}

# ══════════════════════════════════════════════════════════════════════════════
# Combined lookup: all known names from every schema → GarmentCategory
# ══════════════════════════════════════════════════════════════════════════════

ALL_NAME_TO_CATEGORY: dict[str, GarmentCategory] = {
    **COCO_NAME_TO_CATEGORY,
    **DEEPFASHION2_NAME_TO_CATEGORY,
    **FASHIONPEDIA_NAME_TO_CATEGORY,
    **APP_NAME_TO_CATEGORY,          # app-native overrides everything else
}

# ── Keyword-based fallback groups ─────────────────────────────────────────────
# Used when an exact or partial name lookup fails.

_KEYWORD_MAP: list[tuple[tuple[str, ...], GarmentCategory]] = [
    (("shirt", "top", "blouse", "tee", "sweater", "sweatshirt", "vest", "polo"),
     GarmentCategory.SHIRT),
    (("pant", "trouser", "jean", "short", "legging", "chino", "cargo"),
     GarmentCategory.PANTS),
    (("shoe", "boot", "sneaker", "sandal", "slipper", "trainer", "heel", "loafer"),
     GarmentCategory.SHOES),
    (("jacket", "coat", "outwear", "blazer", "windbreaker", "parka", "anorak",
      "cardigan", "cape"),
     GarmentCategory.JACKET),
    (("dress",),
     GarmentCategory.DRESS),
    (("skirt",),
     GarmentCategory.SKIRT),
]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def name_to_category(class_name: str) -> GarmentCategory:
    """
    Map any fashion class name string → ``GarmentCategory``.

    Resolution order:
    1. Exact case-insensitive lookup in ALL_NAME_TO_CATEGORY.
    2. Substring / partial-name lookup in ALL_NAME_TO_CATEGORY.
    3. Keyword group fallback.
    4. ``GarmentCategory.OTHER`` if nothing matches.

    Examples::

        name_to_category("shirt")               → SHIRT
        name_to_category("Long Sleeve Top")     → SHIRT
        name_to_category("denim jeans")         → PANTS
        name_to_category("oxford shoe")         → SHOES
        name_to_category("banana")              → OTHER
    """
    key = class_name.strip().lower()

    # 1. Exact match
    if key in ALL_NAME_TO_CATEGORY:
        return ALL_NAME_TO_CATEGORY[key]

    # 2. Partial match (known name is a substring of key or vice-versa)
    for known_name, cat in ALL_NAME_TO_CATEGORY.items():
        if known_name in key or key in known_name:
            return cat

    # 3. Keyword group fallback
    for keywords, cat in _KEYWORD_MAP:
        if any(kw in key for kw in keywords):
            return cat

    return GarmentCategory.OTHER


def build_label_map_from_model_names(
    names: dict[int, str],
) -> dict[int, GarmentCategory]:
    """
    Auto-build a ``{class_index: GarmentCategory}`` map from a YOLO model's
    ``model.names`` dict.

    This enables any fine-tuned or community model to work without editing
    ``yolo_detector.py`` — class names are resolved via ``name_to_category``.

    Args:
        names: The ``model.names`` dict from a loaded Ultralytics YOLO model,
               e.g. ``{0: 'shirt', 1: 'pants', 2: 'shoes', ...}``.

    Returns:
        A dict mapping each class index to its ``GarmentCategory``.

    Example::

        from ultralytics import YOLO
        model = YOLO("ml/models/yolov8_fashion.pt")
        label_map = build_label_map_from_model_names(model.names)
        # {0: GarmentCategory.SHIRT, 1: GarmentCategory.PANTS, ...}
    """
    return {idx: name_to_category(name) for idx, name in names.items()}


def is_fashion_model(names: dict[int, str]) -> bool:
    """
    Return True if the loaded model appears to be a fashion-specific model
    (i.e. its class names include at least one clothing term).

    A vanilla COCO model returns False because none of its 80 classes are
    in the fashion keyword list.
    """
    fashion_keywords = {
        "shirt", "t-shirt", "pants", "jeans", "shorts",
        "shoes", "sneakers", "jacket", "coat", "dress",
        "skirt", "top", "trouser", "blouse",
    }
    model_names_lower = {n.lower() for n in names.values()}
    return bool(model_names_lower & fashion_keywords)
