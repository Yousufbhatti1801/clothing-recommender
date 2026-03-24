"""
ml/dataset_validator.py
========================
Reusable dataset validation module for YOLO-format fashion datasets.

Called automatically by scripts/train_fashion_yolo.py before training
starts, so broken datasets are caught early rather than mid-run.

Also importable in pytest — tests/test_dataset_validation.py delegates
all real logic here and only asserts on the returned DatasetReport.

CLI usage
---------
    python3.11 ml/dataset_validator.py --data config/fashion_dataset.yaml
    python3.11 ml/dataset_validator.py --data config/fashion_dataset.yaml --fix
    python3.11 ml/dataset_validator.py --data config/fashion_dataset.yaml --strict

Python API
----------
    from ml.dataset_validator import DatasetValidator

    report = DatasetValidator("config/fashion_dataset.yaml").validate()
    report.raise_if_invalid()   # raises ValueError on any error
    print(report.summary())
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
SPLITS: tuple[str, str] = ("train", "val")
MIN_TRAIN_IMAGES: int = 100
MIN_VAL_IMAGES:   int = 20


# ---------------------------------------------------------------------------
# DatasetReport
# ---------------------------------------------------------------------------

@dataclass
class DatasetReport:
    """Structured result of a full dataset validation run."""

    config_path:       pathlib.Path
    dataset_root:      pathlib.Path
    num_classes:       int
    class_names:       dict[int, str]

    # per-split counts
    image_counts:      dict[str, int]    = field(default_factory=dict)
    label_counts:      dict[str, int]    = field(default_factory=dict)
    annotation_counts: dict[str, int]    = field(default_factory=dict)

    # class distribution  {split: Counter{class_idx: instance_count}}
    class_distribution: dict[str, Counter] = field(default_factory=dict)

    # issues collected during validation
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_images(self) -> int:
        return sum(self.image_counts.values())

    @property
    def total_annotations(self) -> int:
        return sum(self.annotation_counts.values())

    @property
    def is_valid(self) -> bool:
        """True when no errors were found (warnings are allowed)."""
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` listing every error, or return silently."""
        if not self.is_valid:
            raise ValueError(
                f"Dataset validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(f"  ✗ {e}" for e in self.errors)
            )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self, verbose: bool = True) -> str:
        SEP = "═" * 62
        lines: list[str] = [
            SEP,
            "  DATASET VALIDATION REPORT",
            SEP,
            f"  Config  : {self.config_path}",
            f"  Root    : {self.dataset_root}",
            f"  Classes : {self.num_classes}",
            "",
            f"  {'Split':<6}  {'Images':>8}  {'Labels':>8}  {'Annotations':>12}",
            f"  {'─' * 46}",
        ]
        for split in SPLITS:
            lines.append(
                f"  {split:<6}  "
                f"{self.image_counts.get(split, 0):>8,}  "
                f"{self.label_counts.get(split, 0):>8,}  "
                f"{self.annotation_counts.get(split, 0):>12,}"
            )
        lines.append(
            f"  {'TOTAL':<6}  "
            f"{self.total_images:>8,}  "
            f"{'':>8}  "
            f"{self.total_annotations:>12,}"
        )

        if verbose and self.class_distribution.get("train"):
            lines += ["", "  CLASS DISTRIBUTION  (train)", f"  {'─' * 52}"]
            train_counts = self.class_distribution["train"]
            total        = sum(train_counts.values()) or 1
            max_cnt      = max(train_counts.values(), default=1)
            for idx in sorted(train_counts):
                cnt  = train_counts[idx]
                name = self.class_names.get(idx, f"class_{idx}")
                bar  = "█" * int(25 * cnt / max_cnt)
                lines.append(
                    f"  [{idx:2d}] {name:22s}  "
                    f"{cnt:6,}  ({cnt / total:5.1%})  {bar}"
                )

        if self.warnings:
            lines += ["", f"  WARNINGS ({len(self.warnings)})"]
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")

        if self.errors:
            lines += ["", f"  ERRORS ({len(self.errors)})"]
            for e in self.errors:
                lines.append(f"  ✗  {e}")
        else:
            lines += ["", "  ✅  All checks passed"]

        lines.append(SEP)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DatasetValidator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """
    Validates a YOLO-format fashion dataset described by a dataset YAML.

    Parameters
    ----------
    config_path : str | Path
        Path to the dataset YAML, e.g. ``config/fashion_dataset.yaml``.
    max_images_to_verify : int
        Maximum number of images opened with Pillow for corruption checks.
        Caps runtime on very large datasets.
    """

    def __init__(
        self,
        config_path: str | pathlib.Path = "config/fashion_dataset.yaml",
        max_images_to_verify: int = 500,
    ) -> None:
        self.config_path          = pathlib.Path(config_path).resolve()
        self.max_images_to_verify = max_images_to_verify

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> DatasetReport:
        """Run all checks and return a populated :class:`DatasetReport`."""
        cfg, dataset_root, num_classes, class_names = self._load_yaml()

        report = DatasetReport(
            config_path  = self.config_path,
            dataset_root = dataset_root,
            num_classes  = num_classes,
            class_names  = class_names,
        )

        self._check_directories(report, dataset_root)
        # Stop early if root is missing — subsequent checks would all fail
        if not dataset_root.exists():
            return report

        self._check_counts(report, dataset_root)
        self._check_label_integrity(report, dataset_root, num_classes)
        self._check_image_integrity(report, dataset_root)
        self._check_class_distribution(report, dataset_root, num_classes)
        return report

    def validate_and_raise(self) -> DatasetReport:
        """Validate and raise :class:`ValueError` on any error found."""
        report = self.validate()
        report.raise_if_invalid()
        return report

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _load_yaml(self) -> tuple[dict, pathlib.Path, int, dict[int, str]]:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required: pip install pyyaml") from exc

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {self.config_path}"
            )

        cfg = yaml.safe_load(self.config_path.read_text())

        # Resolve dataset root — relative paths are relative to the YAML file
        raw_path     = cfg.get("path", ".")
        dataset_root = pathlib.Path(raw_path)
        if not dataset_root.is_absolute():
            dataset_root = (self.config_path.parent / dataset_root).resolve()

        num_classes = int(cfg.get("nc", 0))

        # class names may be a list or a dict in different YOLO YAML variants
        raw_names = cfg.get("names", {})
        if isinstance(raw_names, list):
            class_names = {i: str(n) for i, n in enumerate(raw_names)}
        else:
            class_names = {int(k): str(v) for k, v in raw_names.items()}

        # Stash cfg for split-path resolution in helper methods
        self._cfg = cfg

        return cfg, dataset_root, num_classes, class_names

    # ------------------------------------------------------------------
    # Directory existence
    # ------------------------------------------------------------------

    def _check_directories(
        self,
        report: DatasetReport,
        root: pathlib.Path,
    ) -> None:
        if not root.exists():
            report.errors.append(
                f"Dataset root not found: {root}\n"
                "    Run a dataset preparation script first:\n"
                "      python3.11 scripts/prepare_fashionpedia.py\n"
                "      python3.11 scripts/prepare_fashion_roboflow.py"
            )
            return

        for split in SPLITS:
            img_dir = self._images_dir(root, split)
            lbl_dir = self._labels_dir(root, split)
            if not img_dir.exists():
                report.errors.append(f"Missing images directory: {img_dir.relative_to(root)}")
            if not lbl_dir.exists():
                report.errors.append(f"Missing labels directory: {lbl_dir.relative_to(root)}")

    # ------------------------------------------------------------------
    # Image / label directory resolution (reads YAML train:/val: keys)
    # ------------------------------------------------------------------

    def _images_dir(self, root: pathlib.Path, split: str) -> pathlib.Path:
        """
        Return the images directory for *split*, respecting the YAML's
        ``train:`` / ``val:`` keys (e.g. ``images/train``).
        """
        cfg = getattr(self, "_cfg", {})
        rel = cfg.get(split, f"images/{split}")
        return root / rel

    def _labels_dir(self, root: pathlib.Path, split: str) -> pathlib.Path:
        """
        Return the labels directory for *split*.

        YOLO convention: replace the first ``images`` path component with
        ``labels``.  E.g. ``root/images/train`` → ``root/labels/train``.
        """
        img_dir = self._images_dir(root, split)
        try:
            rel_parts = list(img_dir.relative_to(root).parts)
        except ValueError:
            rel_parts = list(img_dir.parts)
        for i, part in enumerate(rel_parts):
            if part == "images":
                rel_parts[i] = "labels"
                break
        return root.joinpath(*rel_parts)

    def _images_in(self, root: pathlib.Path, split: str) -> list[pathlib.Path]:
        d = self._images_dir(root, split)
        if not d.exists():
            return []
        imgs: list[pathlib.Path] = []
        for ext in SUPPORTED_IMAGE_EXTS:
            imgs.extend(d.glob(f"*{ext}"))
        return sorted(imgs)

    def _labels_in(self, root: pathlib.Path, split: str) -> list[pathlib.Path]:
        d = self._labels_dir(root, split)
        if not d.exists():
            return []
        return sorted(d.glob("*.txt"))

    def _check_counts(
        self,
        report: DatasetReport,
        root: pathlib.Path,
    ) -> None:
        for split in SPLITS:
            imgs = self._images_in(root, split)
            lbls = self._labels_in(root, split)

            report.image_counts[split] = len(imgs)
            report.label_counts[split] = len(lbls)
            report.annotation_counts[split] = sum(
                len([ln for ln in lbl.read_text().splitlines() if ln.strip()])
                for lbl in lbls
            )

        n_train = report.image_counts.get("train", 0)
        n_val   = report.image_counts.get("val", 0)

        if n_train < MIN_TRAIN_IMAGES:
            report.errors.append(
                f"Train set has only {n_train} images "
                f"(minimum {MIN_TRAIN_IMAGES})"
            )
        if n_val < MIN_VAL_IMAGES:
            report.errors.append(
                f"Val set has only {n_val} images "
                f"(minimum {MIN_VAL_IMAGES})"
            )

        for split in SPLITS:
            n_img = report.image_counts.get(split, 0)
            n_lbl = report.label_counts.get(split, 0)
            if n_img != n_lbl:
                report.warnings.append(
                    f"{split}: {n_img} images vs {n_lbl} labels — "
                    "image/label count mismatch"
                )

    # ------------------------------------------------------------------
    # Label file integrity
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_label(path: pathlib.Path) -> list[tuple[int, list[float]]]:
        rows: list[tuple[int, list[float]]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            import contextlib
            with contextlib.suppress(ValueError, IndexError):
                rows.append((int(parts[0]), [float(v) for v in parts[1:]]))
        return rows

    def _check_label_integrity(
        self,
        report: DatasetReport,
        root: pathlib.Path,
        num_classes: int,
    ) -> None:
        bad_class:     list[str] = []
        bad_coord:     list[str] = []
        orphan_labels: list[str] = []
        empty_labels:  list[str] = []

        for split in SPLITS:
            img_dir = self._images_dir(root, split)
            for lbl in self._labels_in(root, split):
                # label with no corresponding image
                has_img = any(
                    (img_dir / f"{lbl.stem}{ext}").exists()
                    for ext in SUPPORTED_IMAGE_EXTS
                )
                if not has_img:
                    orphan_labels.append(f"{split}/{lbl.name}")
                    continue

                rows = self._parse_label(lbl)
                if not rows:
                    empty_labels.append(f"{split}/{lbl.name}")
                    continue

                for cls_idx, coords in rows:
                    if not (0 <= cls_idx < num_classes):
                        bad_class.append(
                            f"{split}/{lbl.name}: cls={cls_idx} "
                            f"(valid 0–{num_classes - 1})"
                        )
                    for i, v in enumerate(coords):
                        if not (0.0 < v <= 1.0):
                            bad_coord.append(
                                f"{split}/{lbl.name}: coord[{i}]={v:.4f}"
                            )

        if orphan_labels:
            report.warnings.append(
                f"{len(orphan_labels)} label file(s) have no matching image"
            )
        if empty_labels:
            report.warnings.append(
                f"{len(empty_labels)} label file(s) are empty"
            )
        if bad_class:
            report.errors.append(
                f"{len(bad_class)} invalid class index/indices:\n"
                + "\n".join(f"      {b}" for b in bad_class[:5])
                + ("\n      …" if len(bad_class) > 5 else "")
            )
        if bad_coord:
            report.errors.append(
                f"{len(bad_coord)} out-of-range bbox coordinate(s):\n"
                + "\n".join(f"      {b}" for b in bad_coord[:5])
                + ("\n      …" if len(bad_coord) > 5 else "")
            )

    # ------------------------------------------------------------------
    # Image corruption check
    # ------------------------------------------------------------------

    def _check_image_integrity(
        self,
        report: DatasetReport,
        root: pathlib.Path,
    ) -> None:
        try:
            from PIL import Image
        except ImportError:
            report.warnings.append(
                "Pillow not installed — skipping image corruption checks"
            )
            return

        corrupt: list[str] = []
        checked = 0

        for split in SPLITS:
            for img_path in self._images_in(root, split):
                if checked >= self.max_images_to_verify:
                    break
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as exc:
                    corrupt.append(f"{split}/{img_path.name}: {exc}")
                checked += 1

        if corrupt:
            report.errors.append(
                f"{len(corrupt)} corrupted image(s):\n"
                + "\n".join(f"      {c}" for c in corrupt[:5])
                + ("\n      …" if len(corrupt) > 5 else "")
            )

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------

    def _check_class_distribution(
        self,
        report: DatasetReport,
        root: pathlib.Path,
        num_classes: int,
    ) -> None:
        for split in SPLITS:
            counter: Counter = Counter()
            for lbl in self._labels_in(root, split):
                for cls_idx, _ in self._parse_label(lbl):
                    counter[cls_idx] += 1
            report.class_distribution[split] = counter

        # every class must appear at least once in train
        train_classes = set(report.class_distribution.get("train", {}).keys())
        missing = set(range(num_classes)) - train_classes
        if missing:
            report.warnings.append(
                "Train set missing class(es): "
                + ", ".join(
                    report.class_names.get(i, str(i)) for i in sorted(missing)
                )
            )

        # warn on extremely rare classes (< 0.5 % of train annotations)
        train_counts = report.class_distribution.get("train", Counter())
        total        = sum(train_counts.values())
        if total > 0:
            rare = [
                report.class_names.get(i, str(i))
                for i, cnt in train_counts.items()
                if cnt / total < 0.005
            ]
            if rare:
                report.warnings.append(
                    "Rare train class(es) (< 0.5 % of annotations): "
                    + ", ".join(rare)
                )


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------

def auto_fix(report: DatasetReport) -> int:
    """
    Remove empty label files — the only safe automatic fix.

    Returns the number of files removed.
    """
    root    = report.dataset_root
    removed = 0
    for split in SPLITS:
        lbl_dir = root / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            if lbl.stat().st_size == 0:
                lbl.unlink()
                removed += 1
    return removed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO fashion dataset before training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        default="config/fashion_dataset.yaml",
        help="Path to dataset YAML (default: config/fashion_dataset.yaml)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix minor issues such as empty label files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 on any warning)",
    )
    parser.add_argument(
        "--max-verify",
        type=int,
        default=500,
        metavar="N",
        help="Maximum images to open for corruption check (default: 500)",
    )
    args = parser.parse_args()

    validator = DatasetValidator(args.data, max_images_to_verify=args.max_verify)
    report    = validator.validate()

    print(report.summary(verbose=True))

    if args.fix:
        n = auto_fix(report)
        if n:
            print(f"\n  🔧  Auto-fixed: removed {n} empty label file(s)")

    if not report.is_valid:
        sys.exit(1)
    if args.strict and report.warnings:
        print(f"\n  [strict mode] {len(report.warnings)} warning(s) treated as errors")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
