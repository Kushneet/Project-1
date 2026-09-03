#!/usr/bin/env python3
"""Phase 4 — run the BASE (un-finetuned) model over the frozen eval subset.

    python scripts/run_baseline.py
    python scripts/run_baseline.py --limit 5      # quick smoke test

MUST be run before fine-tuning. It establishes the comparison point and
freezes the evaluation subset that Phase 10 will reuse.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    build_balanced_subset,
    run_inference,
    select_eval_subset,
)
from src.evaluate import evaluate_results_file  # noqa: E402
from src.prepare_data import load_jsonl  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

LOG = get_logger("run_baseline")


def print_two_track_summary(payload, cfg, n_images: int, n_rows: int, title: str) -> None:
    """Print Track A and Track B results in a consistent layout."""
    a = payload["track_a_binary"]
    b = payload.get("track_b_defect_type") or {}
    view = a.get("primary_view", "full_test_set")
    primary = a.get(view, a["full_test_set"])

    print("\n" + "=" * 66)
    print(title.center(66))
    print("=" * 66)
    print(f"Model   : {cfg['model']['model_name']}")
    print(f"Images  : {n_images} | prompts: {cfg['baseline']['prompt_ids']} | rows: {n_rows}")

    print("\n--- TRACK A (PRIMARY): binary OK vs Defective ---")
    print(f"reporting view: {view}  ({primary.get('n_ok_images')} OK / "
          f"{primary.get('n_defective_images')} Defective)")
    for key, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                       ("precision_defective", "Precision (Defective)"),
                       ("recall_defective", "Recall (Defective)"),
                       ("f1_defective", "F1 (Defective)"), ("macro_f1", "Macro F1"),
                       ("ok_recall", "OK recall"), ("defective_recall", "Defective recall")]:
        print(f"  {label:<24}{primary.get(key)}")
    print(f"  {'OK recall 95% CI':<24}{primary.get('ok_recall_95ci')}")
    cm = primary.get("confusion_matrix", {})
    print(f"  {'False positives':<24}{cm.get('false_positive_ok_as_defective')}")
    print(f"  {'False negatives':<24}{cm.get('false_negative_defective_as_ok')}")
    print(f"  {'Unparseable':<24}{primary.get('n_unparseable')}")
    if primary.get("small_sample_warning"):
        print(f"  ! {primary['small_sample_warning']}")

    full = a["full_test_set"]
    print(f"\n  full held-out test set: accuracy={full.get('accuracy')} "
          f"macro_f1={full.get('macro_f1')} ok_recall={full.get('ok_recall')}")

    if b:
        fb = b["full_test_set"]
        print("\n--- TRACK B (SECONDARY): 12-class defect type ---")
        print(f"  {'Coverage':<24}{fb.get('coverage')} "
              f"({fb.get('n_valid_class_predictions')}/{fb.get('n_total')} valid predictions)")
        for key, label in [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                           ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1"),
                           ("weighted_f1", "Weighted F1")]:
            print(f"  {label:<24}{fb.get(key)}")
        print("  (metrics on valid predictions only — self-selected, so biased upward)")

    print("\n! " + payload["synthetic_label_caveat"])
    print("=" * 66)


def run_baseline(cfg, loaded=None, limit: int | None = None,
                 subset_size: int | None = None, full_test_set: bool = False):
    """Run the base-model baseline and return (results_df, metrics, metadata).

    ``loaded`` lets a notebook pass an already-loaded model so the ~8 GB of
    weights are not loaded a second time in a subprocess — on a 15 GB T4 that
    would OOM.
    """
    import hashlib
    import platform
    from datetime import datetime, timezone

    import torch
    import transformers

    from src.prompts import EVAL_PROMPTS
    from src.utils import write_json

    set_seed(cfg["project"]["seed"])

    test_examples = load_jsonl(cfg["data"]["test_file"])
    LOG.info("Loaded %d held-out test examples", len(test_examples))

    n = None if full_test_set else (subset_size or cfg["baseline"]["eval_subset_size"])
    subset = select_eval_subset(test_examples, n=n, seed=cfg["project"]["seed"])

    balanced = None
    if cfg["baseline"].get("balanced_subset", True):
        balanced = build_balanced_subset(subset, seed=cfg["project"]["seed"])

    if loaded is None:
        from src.inference import load_model

        LOG.info("BASELINE: loading the PRETRAINED model with NO adapter")
        loaded = load_model(
            model_name=cfg["model"]["model_name"],
            adapter_path=None,                   # <- the whole point of Phase 4
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            min_pixels=cfg["model"].get("min_pixels"),
            max_pixels=cfg["model"].get("max_pixels"),
        )
    if loaded.adapter_path is not None:
        raise RuntimeError(
            "Baseline requires the BASE model, but an adapter is loaded: "
            f"{loaded.adapter_path}"
        )

    meta = {
        "run": "baseline",
        "fine_tuned": False,
        "adapter": None,
        "model_id": cfg["model"]["model_name"],
        "model_class": cfg["model"]["model_class"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(loaded.model.device),
        "dtype": str(next(loaded.model.parameters()).dtype),
        "attn_implementation": cfg["model"]["attn_implementation"],
        "generation": {
            "do_sample": cfg["inference"]["do_sample"],
            "max_new_tokens": cfg["model"]["max_new_tokens"],
            "decoding": "greedy",
        },
        "min_pixels": cfg["model"].get("min_pixels"),
        "max_pixels": cfg["model"].get("max_pixels"),
        "seed": cfg["project"]["seed"],
        "prompt_ids": cfg["baseline"]["prompt_ids"],
        "prompt_sha256_16": {
            pid: hashlib.sha256(EVAL_PROMPTS[pid].encode()).hexdigest()[:16]
            for pid in cfg["baseline"]["prompt_ids"]
        },
        "n_images": len(subset),
        "n_prompts": len(cfg["baseline"]["prompt_ids"]),
        "total_generations": len(subset) * len(cfg["baseline"]["prompt_ids"]),
        "balanced_slice_images": balanced["n"] if balanced else None,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta["gpu_name"] = props.name
        meta["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    write_json(meta, "results/baseline/run_metadata.json")
    LOG.info("device=%s dtype=%s transformers=%s",
             meta["device"], meta["dtype"], meta["transformers_version"])

    started = time.time()
    df = run_inference(
        loaded, subset,
        prompt_ids=cfg["baseline"]["prompt_ids"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
        do_sample=cfg["inference"]["do_sample"],
        results_csv=cfg["baseline"]["results_csv"],
        raw_jsonl=cfg["baseline"]["raw_outputs"],
        tag="base",
        limit=limit,
    )
    elapsed = time.time() - started

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["total_inference_seconds"] = round(elapsed, 2)
    meta["avg_generation_seconds"] = round(elapsed / max(len(df), 1), 3)
    write_json(meta, "results/baseline/run_metadata.json")

    payload = evaluate_results_file(
        cfg["baseline"]["results_csv"], "results/baseline", tag="baseline",
        balanced_subset="results/baseline/balanced_subset.json" if balanced else None,
    )
    return df, payload, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run N images (smoke test; NOT for reported results)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Override baseline.eval_subset_size")
    parser.add_argument("--full-test-set", action="store_true",
                        help="Evaluate the entire held-out test split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df, payload, meta = run_baseline(
        cfg, loaded=None, limit=args.limit,
        subset_size=args.subset_size, full_test_set=args.full_test_set,
    )
    print_two_track_summary(payload, cfg, meta["n_images"], len(df),
                            title="BASELINE RESULTS (base model, NOT fine-tuned)")
    print(f"\nTotal inference: {meta['total_inference_seconds']/60:.1f} min "
          f"({meta['avg_generation_seconds']:.2f} s/generation)")


if __name__ == "__main__":
    main()
