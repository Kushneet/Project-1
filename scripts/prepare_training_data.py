#!/usr/bin/env python3
"""Phases 6-7 — build leakage-safe splits and multimodal training data.

    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --inspect-groups   # verify grouping

Requires scripts/analyze_dataset.py to have run first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_analysis import find_dataset_root  # noqa: E402
from src.dataset import (  # noqa: E402
    attach_content_groups,
    build_index,
    drop_exact_duplicates,
    make_splits,
    save_splits,
    split_statistics,
    verify_no_leakage,
)
from src.prepare_data import prepare  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, set_seed, write_json  # noqa: E402

LOG = get_logger("prepare_training_data")


def inspect_groups(df, limit: int = 15) -> None:
    """Print how images were grouped so the heuristic can be eyeballed."""
    sizes = df.groupby("group_key").size().sort_values(ascending=False)
    print("\n" + "=" * 64)
    print("GROUP INSPECTION — verify these look like real source castings")
    print("=" * 64)
    print(f"Images                : {len(df)}")
    print(f"Distinct groups       : {df['group_key'].nunique()}")
    print(f"Mean images per group : {len(df) / max(df['group_key'].nunique(), 1):.2f}")
    print(f"Largest group size    : {sizes.iloc[0] if len(sizes) else 0}")
    print(f"Singleton groups      : {(sizes == 1).sum()}")
    print(f"\nTop {limit} groups by size:")
    for key, n in sizes.head(limit).items():
        labels = sorted(df[df["group_key"] == key]["class_label"].unique())
        print(f"  group {str(key):<16} n={n:<4} classes={labels}")
    multi = sizes[sizes > 1]
    print(f"\nGroups with >1 image  : {len(multi)}  covering {int(multi.sum())} images")
    if "filename_group_key" in df.columns:
        print("Grouping source       : IMAGE CONTENT (near-duplicate clustering)")
        agree = (df.groupby("filename_group_key")["group_key"].nunique() == 1).mean()
        print(f"Filename-heuristic agreement: {100*agree:.1f}% "
              "(low agreement confirms filenames do not encode the source casting)")
    else:
        print("Grouping source       : FILENAME heuristic")
        print("If every group has exactly 1 image, the heuristic did NOT match this")
        print("dataset's filenames and the leakage guarantee is WEAK.")
    for f in df["filename"].head(5):
        print(f"  example filename: {f}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--metadata",
                        default="results/dataset_analysis/image_metadata.csv")
    parser.add_argument("--inspect-groups", action="store_true",
                        help="Print grouping diagnostics and exit without writing")
    parser.add_argument("--defect-type", choices=["auto", "yes", "no"], default="auto",
                        help="Include defect-type supervision (auto: yes if >2 classes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    set_seed(seed)

    df = build_index(args.metadata)
    LOG.info("Indexed %d valid images across %d classes",
             len(df), df["class_label"].nunique())

    sp_cfg = cfg["split"]
    if sp_cfg.get("group_by", "content") == "content":
        df = attach_content_groups(
            df,
            image_root=find_dataset_root(cfg["data"]["raw_dir"]),
            threshold=sp_cfg.get("similarity_threshold", 0.95),
            size=sp_cfg.get("feature_size", 64),
        )

    if args.inspect_groups:
        inspect_groups(df)
        return

    df, dup_report = drop_exact_duplicates(df)
    LOG.info("After duplicate removal: %d images (%s)", len(df), dup_report)

    sp = sp_cfg
    df = make_splits(
        df,
        train_ratio=sp["train_ratio"],
        validation_ratio=sp["validation_ratio"],
        test_ratio=sp["test_ratio"],
        seed=sp["seed"],
    )
    leak = verify_no_leakage(df)
    LOG.info("Leakage check passed: %s", leak)

    save_splits(df, cfg["data"]["splits_dir"])

    n_classes = df["class_label"].nunique()
    include_dt = n_classes > 2 if args.defect_type == "auto" else args.defect_type == "yes"
    LOG.info("Defect-type supervision: %s (%d classes present)", include_dt, n_classes)

    stats = prepare(
        splits_csv=resolve_path(cfg["data"]["splits_dir"]) / "splits.csv",
        image_root=find_dataset_root(cfg["data"]["raw_dir"]),
        out_dir=cfg["data"]["processed_dir"],
        stats_path="results/training/dataset_statistics.json",
        include_defect_type=include_dt,
        seed=seed,
    )
    stats["duplicate_removal"] = dup_report
    stats["leakage_check"] = leak
    stats["split_statistics"] = split_statistics(df)
    stats["seed"] = seed
    stats["split_ratios"] = {
        "train": sp["train_ratio"], "validation": sp["validation_ratio"],
        "test": sp["test_ratio"],
    }
    write_json(stats, "results/training/dataset_statistics.json")

    print("\n" + "=" * 60)
    print("TRAINING DATA PREPARED")
    print("=" * 60)
    print(f"Seed: {seed} | strategy: group split by source casting")
    for name in ("train", "validation", "test"):
        s = stats[name]
        print(f"{name:<11}: {s['n_examples']:>5} examples | "
              f"{s['n_unique_groups']:>4} groups | {s['binary_distribution']}")
    print(f"\nDefect-type supervision enabled: {include_dt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
