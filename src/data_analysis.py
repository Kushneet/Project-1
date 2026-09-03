"""Phase 1 — dataset discovery and analysis.

Nothing here assumes a folder layout. The dataset root is scanned, the class
directories are *inferred* from where the image files actually live, and every
statistic is computed from the files on disk.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from .utils import get_logger, resolve_path, write_json

LOG = get_logger("data_analysis")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Directory names that describe a split rather than a class label.
SPLIT_TOKENS = {"train", "training", "val", "valid", "validation", "test", "testing"}


@dataclass
class ImageRecord:
    """One image on disk plus everything we can learn about it cheaply."""

    path: Path
    relpath: str
    filename: str
    class_label: str
    split_hint: str
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    mode: str | None = None
    image_format: str | None = None
    file_size_bytes: int = 0
    content_hash: str | None = None
    is_valid: bool = True
    error: str | None = None


def find_dataset_root(base: str | Path) -> Path:
    """Return the deepest single-child directory chain under ``base``.

    Kaggle zips often unpack into a redundant wrapper folder; this walks
    through wrappers that contain exactly one subdirectory and no images.
    """
    root = resolve_path(base)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Run: python scripts/download_dataset.py --which all"
        )
    while True:
        children = [c for c in root.iterdir() if not c.name.startswith(".")]
        subdirs = [c for c in children if c.is_dir()]
        has_images = any(c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in children)
        if len(subdirs) == 1 and not has_images:
            root = subdirs[0]
            continue
        return root


def iter_image_files(root: Path) -> Iterable[Path]:
    """Yield every image file under ``root``, skipping hidden/system files."""
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def describe_tree(root: Path, max_depth: int = 3) -> dict[str, Any]:
    """Summarise the directory tree: per-directory file and image counts."""
    tree: dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_dir() or p.name.startswith("."):
            continue
        rel = p.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        files = [c for c in p.iterdir() if c.is_file() and not c.name.startswith(".")]
        tree[str(rel)] = {
            "n_files": len(files),
            "n_images": sum(1 for c in files if c.suffix.lower() in IMAGE_EXTS),
            "n_subdirs": sum(1 for c in p.iterdir() if c.is_dir()),
            "example_files": [c.name for c in files[:3]],
        }
    return tree


def infer_label_and_split(relpath: Path) -> tuple[str, str]:
    """Infer (class_label, split_hint) from an image's path components.

    The class is the last directory component that is not a split keyword;
    the split hint is the first component that *is* one (else "none").
    """
    parts = list(relpath.parts[:-1])  # drop the filename
    split = "none"
    label_parts = []
    for part in parts:
        if part.lower() in SPLIT_TOKENS:
            if split == "none":
                split = part.lower()
        else:
            label_parts.append(part)
    label = label_parts[-1] if label_parts else "__root__"
    return label, split


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 of raw file bytes — used for exact-duplicate detection."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def scan_images(root: Path, compute_hashes: bool = True) -> list[ImageRecord]:
    """Open every image, record its properties, and flag corrupted files."""
    records: list[ImageRecord] = []
    for path in iter_image_files(root):
        rel = path.relative_to(root)
        label, split = infer_label_and_split(rel)
        rec = ImageRecord(
            path=path,
            relpath=str(rel),
            filename=path.name,
            class_label=label,
            split_hint=split,
            file_size_bytes=path.stat().st_size,
        )
        try:
            with Image.open(path) as im:
                im.verify()  # cheap integrity check
            with Image.open(path) as im:
                rec.width, rec.height = im.size
                rec.mode = im.mode
                rec.image_format = im.format
                rec.channels = len(im.getbands())
        except Exception as exc:  # noqa: BLE001 - we want to record any failure
            rec.is_valid = False
            rec.error = f"{type(exc).__name__}: {exc}"
            LOG.warning("Corrupted/unreadable image: %s (%s)", rel, exc)

        if compute_hashes and rec.is_valid:
            try:
                rec.content_hash = _hash_file(path)
            except OSError as exc:  # noqa: BLE001
                LOG.warning("Could not hash %s: %s", rel, exc)

        records.append(rec)
    return records


def records_to_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    """Convert scan results into a tidy DataFrame (one row per image)."""
    rows = []
    for r in records:
        rows.append(
            {
                "relpath": r.relpath,
                "filename": r.filename,
                "class_label": r.class_label,
                "split_hint": r.split_hint,
                "width": r.width,
                "height": r.height,
                "channels": r.channels,
                "mode": r.mode,
                "image_format": r.image_format,
                "file_size_bytes": r.file_size_bytes,
                "content_hash": r.content_hash,
                "is_valid": r.is_valid,
                "error": r.error,
            }
        )
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> dict[str, Any]:
    """Find duplicate filenames and byte-identical image content."""
    name_counts = Counter(df["filename"])
    dup_names = {k: v for k, v in name_counts.items() if v > 1}

    valid = df[df["is_valid"] & df["content_hash"].notna()]
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for h, rel in zip(valid["content_hash"], valid["relpath"]):
        hash_groups[h].append(rel)
    dup_content = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    # Duplicates that straddle a class boundary are label contradictions.
    cross_class = []
    label_by_rel = dict(zip(df["relpath"], df["class_label"]))
    for h, paths in dup_content.items():
        labels = {label_by_rel[p] for p in paths}
        if len(labels) > 1:
            cross_class.append({"hash": h, "labels": sorted(labels), "paths": paths})

    return {
        "n_duplicate_filenames": len(dup_names),
        "duplicate_filename_examples": dict(list(dup_names.items())[:20]),
        "n_exact_duplicate_groups": len(dup_content),
        "n_images_in_duplicate_groups": sum(len(v) for v in dup_content.values()),
        "exact_duplicate_examples": {h: p for h, p in list(dup_content.items())[:20]},
        "n_cross_class_duplicate_groups": len(cross_class),
        "cross_class_duplicates": cross_class[:20],
    }


def summarize(df: pd.DataFrame, root: Path, tree: dict[str, Any]) -> dict[str, Any]:
    """Build the machine-readable dataset summary."""
    valid = df[df["is_valid"]]
    class_counts = valid["class_label"].value_counts().to_dict()
    total_valid = int(len(valid))

    dims = valid.dropna(subset=["width", "height"])
    dim_pairs = Counter(zip(dims["width"].astype(int), dims["height"].astype(int)))

    return {
        "dataset_root": str(root),
        "n_files_total": int(len(df)),
        "n_valid_images": total_valid,
        "n_corrupted_images": int((~df["is_valid"]).sum()),
        "corrupted_examples": df[~df["is_valid"]]["relpath"].head(20).tolist(),
        "class_names": sorted(class_counts),
        "n_classes": len(class_counts),
        "images_per_class": dict(sorted(class_counts.items())),
        "class_percentages": {
            k: round(100.0 * v / total_valid, 3) for k, v in sorted(class_counts.items())
        } if total_valid else {},
        "split_hints": valid["split_hint"].value_counts().to_dict(),
        "image_formats": valid["image_format"].value_counts().to_dict(),
        "channel_distribution": {
            str(k): int(v) for k, v in valid["channels"].value_counts().items()
        },
        "mode_distribution": valid["mode"].value_counts().to_dict(),
        "dimension_stats": {
            "width": {k: float(v) for k, v in dims["width"].describe().items()},
            "height": {k: float(v) for k, v in dims["height"].describe().items()},
        } if len(dims) else {},
        "most_common_dimensions": [
            {"width": int(w), "height": int(h), "count": int(c)}
            for (w, h), c in dim_pairs.most_common(10)
        ],
        "n_distinct_dimensions": len(dim_pairs),
        "file_size_mb_total": round(float(df["file_size_bytes"].sum()) / 1e6, 2),
        "duplicates": duplicate_report(df),
        "directory_tree": tree,
        "label_type": "multi-class" if len(class_counts) > 2 else "binary",
    }


def analyze(
    data_dir: str | Path,
    out_dir: str | Path,
    compute_hashes: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full Phase-1 analysis and persist CSV + JSON artefacts."""
    root = find_dataset_root(data_dir)
    LOG.info("Resolved dataset root: %s", root)

    tree = describe_tree(root)
    records = scan_images(root, compute_hashes=compute_hashes)
    if not records:
        raise RuntimeError(f"No image files found under {root}")
    LOG.info("Scanned %d image files", len(records))

    df = records_to_dataframe(records)
    summary = summarize(df, root, tree)

    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "image_metadata.csv", index=False)
    write_json(summary, out / "dataset_summary.json")
    LOG.info("Wrote %s and %s", out / "image_metadata.csv", out / "dataset_summary.json")
    return df, summary
