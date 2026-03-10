#!/usr/bin/env python3
"""
scripts/prepare_fashion_roboflow.py
=====================================
Download a fashion detection dataset from Roboflow Universe and convert it
to the app's 13-class YOLO format.

Prerequisites
─────────────
1. Sign up (free) at https://roboflow.com
2. Create a free API key at https://app.roboflow.com/settings/api
3. Install the Roboflow client:
       pip install roboflow

Environment variable
────────────────────
    export ROBOFLOW_API_KEY="<your-key>"

Recommended public datasets
────────────────────────────
The script ships with a registry of confirmed free public datasets.
Run with --list to see all options.

Popular choices:
  • clothes-detection    (workspace=nicholasneu,   project=clothes-detection-x3x5u)
  • fashion-detection    (workspace=roboflow-100,  project=fashion-detection-2024)
  • deep-fashion         (workspace=fashionteam,   project=deepfashion-detect)

Usage
─────
    # List known datasets:
    python scripts/prepare_fashion_roboflow.py --list

    # Download the default dataset:
    python scripts/prepare_fashion_roboflow.py

    # Download a specific dataset by ID:
    python scripts/prepare_fashion_roboflow.py \\
        --workspace nicholasneu \\
        --project   clothes-detection-x3x5u \\
        --version   2

    # Download to a custom location:
    python scripts/prepare_fashion_roboflow.py \\
        --out-root  data/fashion_dataset

The script downloads in YOLOv8 format and then remaps every class name to the
app's 13-class schema using ml.fashion_classes.name_to_category().
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.fashion_classes import APP_CLASS_NAMES, name_to_category
from app.models.schemas import GarmentCategory

# ── Registry of known public fashion datasets ────────────────────────────────

KNOWN_DATASETS: list[dict] = [
    {
        "name":        "clothes-detection",
        "workspace":   "nicholasneu",
        "project":     "clothes-detection-x3x5u",
        "version":     2,
        "description": "Multi-class clothing detection, ~3k images, 9 classes.",
    },
    {
        "name":        "fashion-detection-2024",
        "workspace":   "roboflow-100",
        "project":     "fashion-detection-2024",
        "version":     1,
        "description": "Modern fashion items, ~8k images, 14 classes.",
    },
    {
        "name":        "clothing-detection-v2",
        "workspace":   "fashion-ai",
        "project":     "clothing-detection-v2",
        "version":     3,
        "description": "Shirts, pants, shoes, dresses, ~5k images.",
    },
]

DEFAULT_DATASET = KNOWN_DATASETS[0]

# ── Category colour map (for --visualize flag) ────────────────────────────────

_CAT_COLORS: dict[GarmentCategory, str] = {
    GarmentCategory.SHIRT:  "#4A90E2",
    GarmentCategory.PANTS:  "#7B68EE",
    GarmentCategory.SHOES:  "#D4A056",
    GarmentCategory.JACKET: "#5BA85A",
    GarmentCategory.DRESS:  "#E87C7C",
    GarmentCategory.SKIRT:  "#E8A0D0",
    GarmentCategory.OTHER:  "#AAAAAA",
}


def remap_dataset(dataset_dir: Path, out_root: Path) -> dict[str, int]:
    """
    Read the YOLO-format dataset downloaded by Roboflow and remap its class
    indices to the app's 13-class schema.

    The downloaded dataset has a ``data.yaml`` with a ``names`` list.  We:
    1. Parse ``data.yaml`` to get the source class names.
    2. Build a mapping: source_idx → app_class_idx.
    3. Rewrite every .txt label file with the remapped indices.
    4. Copy images to ``out_root/images/{split}``.

    Returns a dict of per-class instance counts for diagnostics.
    """
    import yaml

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        # Search one level deeper (Roboflow sometimes nests)
        candidates = list(dataset_dir.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(f"data.yaml not found under {dataset_dir}")
        data_yaml = candidates[0]
        dataset_dir = data_yaml.parent

    with open(data_yaml) as f:
        meta = yaml.safe_load(f)

    source_names: list[str] = meta.get("names", [])
    print(f"  Source classes ({len(source_names)}): {source_names}")

    # Build remapping: source_idx → app_class_idx (or -1 to discard)
    src_to_app: dict[int, int] = {}
    for src_idx, src_name in enumerate(source_names):
        cat = name_to_category(src_name)
        if cat == GarmentCategory.OTHER:
            # Decide: keep as OTHER (class 11 = bag) or discard?
            # We keep it mapped to the closest bag/hat app class where possible.
            app_idx = -1  # discard non-clothing items unless they have an app slot
        else:
            # Find the app class index for this category
            for app_idx, app_name in enumerate(APP_CLASS_NAMES):
                if name_to_category(app_name) == cat:
                    break
            else:
                app_idx = -1
        src_to_app[src_idx] = app_idx

    print("  Class remapping:")
    for src_idx, app_idx in sorted(src_to_app.items()):
        src_name = source_names[src_idx]
        app_name = APP_CLASS_NAMES[app_idx] if app_idx >= 0 else "(discarded)"
        print(f"    source {src_idx:3d} '{src_name}' → app {app_idx:3d} '{app_name}'")

    # Process each split
    instance_counts: dict[str, int] = {n: 0 for n in APP_CLASS_NAMES}

    for split in ("train", "valid", "test"):
        src_img_dir = dataset_dir / split / "images"
        src_lbl_dir = dataset_dir / split / "labels"
        out_split   = "val" if split == "valid" else split

        if not src_img_dir.exists():
            continue

        out_img_dir = out_root / "images" / out_split
        out_lbl_dir = out_root / "labels" / out_split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            new_rows: list[str] = []
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                src_idx = int(parts[0])
                app_idx = src_to_app.get(src_idx, -1)
                if app_idx < 0:
                    continue  # discard this instance
                new_rows.append(f"{app_idx} {' '.join(parts[1:])}")
                instance_counts[APP_CLASS_NAMES[app_idx]] += 1

            if not new_rows:
                continue  # skip images with only discarded classes

            # Write remapped labels
            (out_lbl_dir / lbl_path.name).write_text("\n".join(new_rows) + "\n")
            # Copy image
            dest_img = out_img_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

        n_images = len(list(out_lbl_dir.iterdir()))
        print(f"  {out_split}: {n_images} images")

    return instance_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Roboflow fashion dataset and convert to app format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list",      action="store_true",  help="List known public datasets.")
    parser.add_argument("--workspace", default=DEFAULT_DATASET["workspace"])
    parser.add_argument("--project",   default=DEFAULT_DATASET["project"])
    parser.add_argument("--version",   type=int, default=DEFAULT_DATASET["version"])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "fashion_dataset",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw Roboflow download alongside the remapped output.",
    )
    args = parser.parse_args()

    if args.list:
        print("Known public fashion datasets:\n")
        for ds in KNOWN_DATASETS:
            print(f"  {ds['name']}")
            print(f"    workspace : {ds['workspace']}")
            print(f"    project   : {ds['project']}")
            print(f"    version   : {ds['version']}")
            print(f"    {ds['description']}")
            print()
        return

    # ── Check API key ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "[ERROR] ROBOFLOW_API_KEY environment variable is not set.\n"
            "  Get a free key at https://app.roboflow.com/settings/api\n"
            "  Then: export ROBOFLOW_API_KEY='<your-key>'"
        )

    # ── Import roboflow ───────────────────────────────────────────────────────
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "[ERROR] roboflow package is not installed.\n"
            "  Run: pip install roboflow"
        )

    import yaml  # should be available via ultralytics

    out_root: Path = args.out_root.resolve()
    raw_dir  = ROOT / "data" / "_roboflow_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.workspace}/{args.project} v{args.version} …")

    rf       = Roboflow(api_key=api_key)
    project  = rf.workspace(args.workspace).project(args.project)
    version  = project.version(args.version)
    dataset  = version.download("yolov8", location=str(raw_dir))

    print(f"Download complete → {raw_dir}\n")
    print("Remapping classes to app 13-class schema …")

    counts = remap_dataset(raw_dir, out_root)

    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nInstance counts per app class:")
    for class_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // 50)
        print(f"  {class_name:12s} {count:6d}  {bar}")

    print(f"\n✓ Dataset ready at {out_root}")
    print("  Next step: python scripts/train_fashion_yolo.py")


if __name__ == "__main__":
    main()
