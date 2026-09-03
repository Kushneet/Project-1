"""Dataset indexing and leakage-safe splitting (Phase 7).

Why a *group* split
-------------------
In the primary dataset the defect classes were synthesised by painting defects
onto real OK castings, and hand overlays were then composited on top. One
source casting therefore appears many times: as an ``ok/`` image and as several
defective derivatives. A plain stratified split would place a casting in train
and its own derivative in test, leaking the answer and inflating accuracy.

We therefore split by *group* (source casting), so every derivative of a given
casting lands in exactly one split.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import get_logger, resolve_path, write_json

LOG = get_logger("dataset")

# Suffixes appended by the dataset's augmentation pipeline. Stripping them
# recovers the underlying source-casting identifier.
_AUG_SUFFIX_PAT = re.compile(
    r"(_?(?:hand|overlay|aug|augmented|var|variant|pos|position|rand|random|"
    r"copy|sample|img|image)?[_-]?\d+)+$",
    re.I,
)
_ID_PAT = re.compile(r"(\d{3,})")

OK_TOKENS = {"ok", "ok_front", "good", "normal", "no_defect", "non_defective", "okay"}


def is_defective_label(label: str) -> bool:
    """Map a class-directory name to the binary OK/Defective target.

    Anything not explicitly an OK token counts as defective. Kept strict and
    explicit so an unexpected folder name is never silently treated as OK.
    """
    return label.strip().lower().replace("-", "_") not in OK_TOKENS


def derive_group_key(relpath: str, filename: str) -> str:
    """Infer the source-casting identifier for one image.

    Heuristic, in order of preference:
      1. the longest run of digits in the stem (the pipeline's casting index)
      2. the stem with trailing augmentation suffixes stripped

    IMPORTANT: verify this against the real filenames after download with
    ``python scripts/prepare_training_data.py --inspect-groups``. If the
    grouping is wrong the leakage guarantee does not hold.
    """
    stem = Path(filename).stem
    ids = _ID_PAT.findall(stem)
    if ids:
        return max(ids, key=len)
    stripped = _AUG_SUFFIX_PAT.sub("", stem).strip("_-")
    return stripped or stem


def build_index(metadata_csv: str | Path) -> pd.DataFrame:
    """Load image_metadata.csv and attach label/group columns."""
    path = resolve_path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/analyze_dataset.py first."
        )
    df = pd.read_csv(path)
    df = df[df["is_valid"]].copy()
    if df.empty:
        raise RuntimeError(f"No valid images listed in {path}")

    df["defect_type"] = df["class_label"]
    df["is_defective"] = df["class_label"].map(is_defective_label)
    df["binary_label"] = df["is_defective"].map({True: "Defective", False: "OK"})
    df["group_key"] = [
        str(derive_group_key(r, f)) for r, f in zip(df["relpath"], df["filename"])
    ]
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove byte-identical duplicate images, keeping the first occurrence.

    Cross-class duplicates (same bytes, contradictory labels) are dropped
    entirely — they cannot be labelled correctly and would corrupt both
    training and evaluation.
    """
    report: dict[str, Any] = {"n_before": int(len(df))}
    if "content_hash" not in df.columns or df["content_hash"].isna().all():
        report["note"] = "no content hashes available; duplicate removal skipped"
        report["n_after"] = int(len(df))
        return df, report

    labels_per_hash = df.groupby("content_hash")["binary_label"].nunique()
    contradictory = set(labels_per_hash[labels_per_hash > 1].index)
    if contradictory:
        LOG.warning(
            "Dropping %d images in %d cross-class duplicate groups (contradictory labels)",
            int(df["content_hash"].isin(contradictory).sum()), len(contradictory),
        )
    kept = df[~df["content_hash"].isin(contradictory)].copy()

    before = len(kept)
    kept = kept.drop_duplicates(subset="content_hash", keep="first")
    report.update(
        n_cross_class_groups_dropped=len(contradictory),
        n_exact_duplicates_removed=int(before - len(kept)),
        n_after=int(len(kept)),
    )
    return kept, report


def _greedy_group_split(
    groups: list[str],
    group_sizes: dict[str, int],
    group_label: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to train/val/test, balancing class proportions.

    Groups are shuffled deterministically, then greedily placed into whichever
    split is furthest below its target quota for that group's majority class.
    """
    import random

    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)

    names = ("train", "validation", "test")
    total = sum(group_sizes.values())
    targets = {n: r * total for n, r in zip(names, ratios)}
    # Track filled counts per split per class so both stay proportional.
    filled: dict[str, Counter] = {n: Counter() for n in names}
    class_totals = Counter()
    for g in order:
        class_totals[group_label[g]] += group_sizes[g]

    assignment: dict[str, str] = {}
    for g in order:
        cls = group_label[g]
        best, best_deficit = names[0], -1e18
        for n, r in zip(names, ratios):
            quota = r * class_totals[cls]
            deficit = quota - filled[n][cls]
            # Prefer the split with the largest unmet quota for this class.
            if deficit > best_deficit:
                best, best_deficit = n, deficit
        assignment[g] = best
        filled[best][cls] += group_sizes[g]
    _ = targets  # retained for readability of intent
    return assignment


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "group_key",
) -> pd.DataFrame:
    """Add a ``split`` column using a reproducible, leakage-safe group split."""
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    sizes = df.groupby(group_by).size().to_dict()
    # A group's label is its majority binary label.
    label = (
        df.groupby(group_by)["binary_label"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    assignment = _greedy_group_split(
        list(sizes), sizes, label, (train_ratio, validation_ratio, test_ratio), seed
    )
    out = df.copy()
    out["split"] = out[group_by].map(assignment)
    return out


def verify_no_leakage(df: pd.DataFrame, group_by: str = "group_key") -> dict[str, Any]:
    """Assert that no group and no image hash spans two splits.

    Raises RuntimeError on violation — a failed split must never be used.
    """
    spans = df.groupby(group_by)["split"].nunique()
    leaked_groups = spans[spans > 1].index.tolist()

    leaked_hashes: list[str] = []
    if "content_hash" in df.columns and df["content_hash"].notna().any():
        h = df.dropna(subset=["content_hash"]).groupby("content_hash")["split"].nunique()
        leaked_hashes = h[h > 1].index.tolist()

    if leaked_groups or leaked_hashes:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {len(leaked_groups)} groups and "
            f"{len(leaked_hashes)} image hashes span multiple splits. "
            f"Examples: {leaked_groups[:5]}"
        )
    return {
        "leakage_free": True,
        "n_groups": int(df[group_by].nunique()),
        "checked_hashes": bool(leaked_hashes == [] and "content_hash" in df.columns),
    }


def split_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Per-split counts, class distribution and group counts."""
    stats: dict[str, Any] = {"total_images": int(len(df))}
    for split in ("train", "validation", "test"):
        sub = df[df["split"] == split]
        stats[split] = {
            "n_images": int(len(sub)),
            "n_groups": int(sub["group_key"].nunique()),
            "binary_distribution": sub["binary_label"].value_counts().to_dict(),
            "defect_type_distribution": sub["defect_type"].value_counts().to_dict(),
        }
    return stats


def save_splits(df: pd.DataFrame, splits_dir: str | Path) -> Path:
    """Persist the split assignment as CSV plus a statistics JSON."""
    out = resolve_path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "relpath", "filename", "class_label", "defect_type", "binary_label",
        "is_defective", "group_key", "split", "content_hash",
    ]
    df[[c for c in cols if c in df.columns]].to_csv(out / "splits.csv", index=False)
    write_json(split_statistics(df), out / "split_statistics.json")
    LOG.info("Wrote %s", out / "splits.csv")
    return out / "splits.csv"


# ---------------------------------------------------------------------------
# Content-based grouping
# ---------------------------------------------------------------------------
#
# The primary dataset's filenames are per-class counters (``crack_00042.jpg``)
# and carry NO source-casting identity: measured index-match rate between a
# defect image and the same-index OK image is 2.9%, i.e. chance. Grouping by
# filename would therefore produce arbitrary groups and a false sense of
# safety. Source castings *are* reused across classes (hundreds of cross-class
# pairs exceed 0.99 normalised correlation), so groups must be recovered from
# image content instead.

DEFAULT_SIMILARITY_THRESHOLD = 0.95
DEFAULT_FEATURE_SIZE = 64


def compute_visual_features(
    paths: list[Path], size: int = DEFAULT_FEATURE_SIZE
) -> "np.ndarray":
    """Return L2-normalised, contrast-normalised thumbnails as feature vectors.

    Per-image mean/std normalisation removes exposure and contrast differences,
    so two photographs of the same casting match even under a global brightness
    shift (which several of the synthetic defect transforms introduce).
    """
    import numpy as np
    from PIL import Image

    vectors = []
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(
                im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32
            )
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        vectors.append(arr.ravel())
    X = np.stack(vectors)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def compute_content_groups(
    df: pd.DataFrame,
    image_root: str | Path,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    size: int = DEFAULT_FEATURE_SIZE,
    max_images: int = 20000,
) -> pd.Series:
    """Group near-duplicate images into connected components.

    Two images are linked when their normalised correlation exceeds
    ``threshold``; each connected component becomes one group, standing in for
    one source casting. A lower threshold groups more aggressively, which is
    the *safe* direction for leakage.

    Returns a Series of group keys aligned to ``df.index``.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    if len(df) > max_images:
        raise RuntimeError(
            f"{len(df)} images exceeds max_images={max_images}; the pairwise "
            "similarity matrix would be too large. Raise max_images only if you "
            "have the memory for an N^2 float matrix."
        )

    root = resolve_path(image_root)
    paths = [root / rel for rel in df["relpath"]]
    LOG.info("Computing visual features for %d images (size=%d)", len(paths), size)
    X = compute_visual_features(paths, size=size)

    LOG.info("Building similarity graph at threshold %.3f", threshold)
    sim = X @ X.T
    np.fill_diagonal(sim, 0.0)
    adjacency = sp.csr_matrix(sim > threshold)
    n_components, labels = connected_components(adjacency, directed=False)

    sizes = np.bincount(labels)
    LOG.info(
        "Found %d content groups (largest=%d, singletons=%d, mean size=%.2f)",
        n_components, sizes.max(), int((sizes == 1).sum()), sizes.mean(),
    )
    return pd.Series([f"g{l:05d}" for l in labels], index=df.index)


def attach_content_groups(
    df: pd.DataFrame,
    image_root: str | Path,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    size: int = DEFAULT_FEATURE_SIZE,
) -> pd.DataFrame:
    """Replace ``group_key`` with content-derived groups, keeping the old key."""
    out = df.copy()
    out["filename_group_key"] = out["group_key"]
    out["group_key"] = compute_content_groups(out, image_root, threshold, size)
    return out
