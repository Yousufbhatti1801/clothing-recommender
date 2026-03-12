#!/usr/bin/env python3
"""
scripts/train_fashion_yolo.py
==============================
Fine-tune YOLOv8 on the app's 13-class fashion dataset and export the
best weights to ``ml/models/yolov8_fashion.pt``.

Prerequisites
─────────────
1. Prepare a dataset using one of:
       python scripts/prepare_deepfashion2.py --df2-root /path/to/deepfashion2
   or
       python scripts/prepare_fashion_roboflow.py

2. Verify the dataset exists:
       data/fashion_dataset/images/train/   (≥ 100 images recommended)
       data/fashion_dataset/images/val/
       data/fashion_dataset/labels/train/
       data/fashion_dataset/labels/val/

Usage
─────
    # Default: yolov8s, 100 epochs, auto device
    python scripts/train_fashion_yolo.py

    # Custom settings:
    python scripts/train_fashion_yolo.py \\
        --model   yolov8m.pt \\
        --epochs  200 \\
        --batch   8 \\
        --device  mps         # 'mps' (Apple M-series), 'cuda', or 'cpu'

    # Resume interrupted training:
    python scripts/train_fashion_yolo.py --resume

    # Quick smoke-test on 5 epochs:
    python scripts/train_fashion_yolo.py --epochs 5 --batch 4

Model sizes (choose based on hardware)
───────────────────────────────────────
    yolov8n.pt   — nano    (~3 MB)   fast, lower accuracy
    yolov8s.pt   — small   (~22 MB)  recommended default
    yolov8m.pt   — medium  (~52 MB)  higher accuracy, needs more VRAM
    yolov8l.pt   — large   (~87 MB)  GPU recommended
    yolov8x.pt   — xlarge  (~130 MB) GPU required

Training time estimates (Apple M2 Pro)
───────────────────────────────────────
    yolov8s, 100 epochs, ~10k images: ~4 h on MPS
    yolov8s, 100 epochs, ~10k images: ~2 h on CUDA RTX 3080
    yolov8n, 100 epochs, ~10k images: ~1.5 h on MPS
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset_validator import DatasetValidator  # noqa: E402

CONFIG_YAML    = ROOT / "config" / "fashion_dataset.yaml"
OUTPUT_WEIGHTS = ROOT / "ml" / "models" / "yolov8_fashion.pt"
RUNS_DIR       = ROOT / "ml" / "runs"

# Validated hyperparameter presets ─────────────────────────────────────────────
# Tuned for fashion detection (small, high-overlap objects, varied lighting).

HPARAMS_FASHION: dict = {
    # Learning rate schedule
    "lr0":            0.01,      # initial learning rate (SGD)
    "lrf":            0.01,      # final lr = lr0 * lrf (cosine annealing)
    "momentum":       0.937,
    "weight_decay":   0.0005,
    # Warmup
    "warmup_epochs":  3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr":  0.1,
    # Loss weights
    "box":   7.5,   # box regression loss
    "cls":   0.5,   # classification loss
    "dfl":   1.5,   # distribution focal loss
    # Augmentation
    "degrees":     0.0,   # rotation (fashion items have fixed orientation)
    "translate":   0.1,
    "scale":       0.5,
    "shear":       0.0,
    "perspective": 0.0,
    "flipud":      0.0,   # fashion images are rarely upside down
    "fliplr":      0.5,
    "mosaic":      1.0,
    "mixup":       0.1,   # blends two training images; helps with layering
    "copy_paste":  0.0,
    # Color augmentation
    "hsv_h":  0.015,
    "hsv_s":  0.7,
    "hsv_v":  0.4,
}


def detect_device() -> str:
    """Auto-select the best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"  # first CUDA GPU
    except ImportError:
        pass
    return "cpu"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for 13-class fashion detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Base model weights (default: yolov8s.pt). "
             "Use yolov8n.pt for fast iteration, yolov8m.pt for best accuracy.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16). Reduce to 8 or 4 if OOM.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'mps', 'cuda', 'cpu', or GPU index (default: auto).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=CONFIG_YAML,
        help=f"Dataset YAML path (default: {CONFIG_YAML}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4). Set 0 for debugging.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience in epochs (default: 50).",
    )
    parser.add_argument(
        "--name",
        default="fashion_yolo",
        help="Experiment name for the runs/ directory (default: fashion_yolo).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of fine-tuning (not recommended).",
    )
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"\n{'='*60}")
    print(f"  Fashion YOLOv8 Fine-tuning")
    print(f"{'='*60}")
    print(f"  Base model : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {device}")
    print(f"  Output     : {OUTPUT_WEIGHTS}")
    print()

    # ── Validate dataset ──────────────────────────────────────────────────────
    try:
        validator = DatasetValidator(args.data)
        report    = validator.validate()
        print(report.summary())
        report.raise_if_invalid()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"[ERROR] {exc}")

    # ── Import ultralytics ────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics is not installed. Run: pip install ultralytics")

    # ── Load base model ───────────────────────────────────────────────────────
    if args.resume:
        # Look for last.pt in runs dir
        last_candidates = sorted(RUNS_DIR.rglob("last.pt"))
        if not last_candidates:
            sys.exit(
                "[ERROR] --resume: no checkpoint found in ml/runs/.\n"
                "  Start a fresh training run first."
            )
        resume_path = last_candidates[-1]
        print(f"  Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        model = YOLO(args.model)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training … (this will take a while)\n")
    t0 = time.perf_counter()

    results = model.train(
        data        = str(args.data.resolve()),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = device,
        project     = str(RUNS_DIR),
        name        = args.name,
        pretrained  = not args.no_pretrained,
        optimizer   = "SGD",
        seed        = 42,
        workers     = args.workers,
        patience    = args.patience,
        save        = True,
        save_period = 10,           # checkpoint every 10 epochs
        plots       = True,         # generate training curve plots
        resume      = args.resume,
        verbose     = True,
        **HPARAMS_FASHION,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f} h")

    # ── Export best weights ───────────────────────────────────────────────────
    # ultralytics saves to runs/<project>/<name>/weights/best.pt
    run_dir   = Path(results.save_dir)
    best_pt   = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        # Fall back to last.pt
        best_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, OUTPUT_WEIGHTS)
        print(f"\n✓ Best weights saved to: {OUTPUT_WEIGHTS}")
    else:
        print(f"\n[WARNING] Could not find best.pt under {run_dir}/weights/")
        print(f"  Manually copy the best weights to {OUTPUT_WEIGHTS}")

    # ── Validation metrics ────────────────────────────────────────────────────
    print("\nFinal validation metrics:")
    try:
        metrics = model.val(data=str(args.data.resolve()), device=device, verbose=False)
        print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"  Precision    : {metrics.box.mp:.4f}")
        print(f"  Recall       : {metrics.box.mr:.4f}")
    except Exception as exc:
        print(f"  (could not compute metrics: {exc})")

    print(f"\nNext step: restart the application to load the new model.")
    print(f"  The detector will auto-detect fashion classes from model.names.")
    print(f"  Run: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
