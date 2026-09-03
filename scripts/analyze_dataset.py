#!/usr/bin/env python3
"""Phase 1 — run dataset analysis and write plots + summary artefacts.

    python scripts/analyze_dataset.py --data-dir data/raw \
        --out-dir results/dataset_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data_analysis import analyze, find_dataset_root  # noqa: E402
from src.utils import get_logger, resolve_path  # noqa: E402

LOG = get_logger("analyze_dataset")


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of image count per class."""
    counts = df[df["is_valid"]]["class_label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(counts))))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Number of images")
    ax.set_title("Class distribution")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_dimensions(df: pd.DataFrame, out: Path) -> None:
    """Scatter of width vs height plus a file-size histogram."""
    valid = df[df["is_valid"]].dropna(subset=["width", "height"])
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(valid["width"], valid["height"], s=8, alpha=0.4, color="#DD8452")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image dimensions")
    axes[1].hist(valid["file_size_bytes"] / 1024, bins=50, color="#55A868")
    axes[1].set_xlabel("File size (KB)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("File size distribution")
    fig.tight_layout()
    fig.savefig(out / "image_dimensions.png", dpi=150)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, root: Path, out: Path, per_class: int = 3) -> None:
    """Grid of representative sample images, one row per class."""
    valid = df[df["is_valid"]]
    classes = sorted(valid["class_label"].unique())
    if not classes:
        return
    fig, axes = plt.subplots(
        len(classes), per_class,
        figsize=(2.4 * per_class, 2.4 * len(classes)),
        squeeze=False,
    )
    for r, cls in enumerate(classes):
        subset = valid[valid["class_label"] == cls].head(per_class)
        for c in range(per_class):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(subset):
                rel = subset.iloc[c]["relpath"]
                try:
                    with Image.open(root / rel) as im:
                        ax.imshow(im, cmap="gray" if im.mode == "L" else None)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("Could not render %s: %s", rel, exc)
            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")
    fig.suptitle("Representative samples per class", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "sample_images.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="results/dataset_analysis")
    parser.add_argument("--no-hashes", action="store_true",
                        help="Skip content hashing (faster, disables duplicate detection)")
    args = parser.parse_args()

    df, summary = analyze(args.data_dir, args.out_dir, compute_hashes=not args.no_hashes)

    out = resolve_path(args.out_dir)
    root = find_dataset_root(args.data_dir)
    plot_class_distribution(df, out)
    plot_dimensions(df, out)
    plot_samples(df, root, out)

    # Console summary — the numbers a human actually wants to see first.
    print("\n" + "=" * 62)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 62)
    print(f"Root                : {summary['dataset_root']}")
    print(f"Files scanned       : {summary['n_files_total']}")
    print(f"Valid images        : {summary['n_valid_images']}")
    print(f"Corrupted images    : {summary['n_corrupted_images']}")
    print(f"Classes ({summary['n_classes']})".ljust(20) + f": {summary['class_names']}")
    print(f"Label type          : {summary['label_type']}")
    print(f"Split hints         : {summary['split_hints']}")
    print(f"Channels            : {summary['channel_distribution']}")
    print(f"Formats             : {summary['image_formats']}")
    print(f"Common dimensions   : {summary['most_common_dimensions'][:3]}")
    print("\nImages per class:")
    for k, v in summary["images_per_class"].items():
        pct = summary["class_percentages"].get(k, 0)
        print(f"  {k:<22} {v:>6}  ({pct:>5.2f}%)")
    dup = summary["duplicates"]
    print("\nDuplicate / leakage check:")
    print(f"  duplicate filenames        : {dup['n_duplicate_filenames']}")
    print(f"  exact-duplicate groups     : {dup['n_exact_duplicate_groups']}")
    print(f"  images in those groups     : {dup['n_images_in_duplicate_groups']}")
    print(f"  cross-class duplicates     : {dup['n_cross_class_duplicate_groups']}")
    print("=" * 62)
    print(f"Artefacts written to {out}")


if __name__ == "__main__":
    main()
