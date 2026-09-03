#!/usr/bin/env python3
"""Phase 10 — evaluate the FINE-TUNED model on the SAME frozen subset.

    python scripts/evaluate_model.py

Reuses results/baseline/eval_subset.json and the identical prompts, so the
comparison against the baseline is like-for-like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    build_balanced_subset,
    run_inference,
    select_eval_subset,
)
from scripts.run_baseline import print_two_track_summary  # noqa: E402
from src.evaluate import evaluate_results_file  # noqa: E402
from src.prepare_data import load_jsonl  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, set_seed  # noqa: E402

LOG = get_logger("evaluate_model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter", default=None, help="Override the adapter path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    adapter = resolve_path(args.adapter or cfg["training"]["output_dir"])
    if not adapter.exists():
        raise SystemExit(
            f"No fine-tuned adapter at {adapter}.\n"
            "Run scripts/train_model.py first (Phase 8)."
        )

    from src.inference import load_model  # imported late: pulls in torch

    test_examples = load_jsonl(cfg["data"]["test_file"])
    # reuse=True is essential: identical images to the baseline.
    subset = select_eval_subset(test_examples, n=None, seed=cfg["project"]["seed"],
                                reuse=True)
    # reuse=True: the identical balanced slice the baseline reported on.
    balanced = build_balanced_subset(subset, seed=cfg["project"]["seed"], reuse=True) \
        if cfg["baseline"].get("balanced_subset", True) else None
    LOG.info("Evaluating on the frozen baseline subset: %d images", len(subset))

    loaded = load_model(
        model_name=cfg["model"]["model_name"],
        adapter_path=adapter,
        dtype=cfg["model"]["dtype"],
        attn_implementation=cfg["model"]["attn_implementation"],
        min_pixels=cfg["model"].get("min_pixels"),
        max_pixels=cfg["model"].get("max_pixels"),
    )

    run_inference(
        loaded, subset,
        prompt_ids=cfg["baseline"]["prompt_ids"],   # identical prompts
        max_new_tokens=cfg["model"]["max_new_tokens"],
        do_sample=cfg["inference"]["do_sample"],
        results_csv=cfg["evaluation"]["results_csv"],
        raw_jsonl=cfg["evaluation"]["raw_outputs"],
        tag="finetuned",
        limit=args.limit,
    )

    payload = evaluate_results_file(
        cfg["evaluation"]["results_csv"], "results/evaluation", tag="finetuned",
        balanced_subset="results/baseline/balanced_subset.json" if balanced else None,
    )
    print_two_track_summary(payload, cfg, len(subset),
                            len(pd.read_csv(resolve_path(cfg["evaluation"]["results_csv"]))),
                            title="FINE-TUNED RESULTS")
    print("Next: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
