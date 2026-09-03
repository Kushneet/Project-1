#!/usr/bin/env python3
"""Phases 8-9 — LoRA fine-tuning, with a mandatory sanity check first.

    python scripts/train_model.py --sanity-check    # Phase 9, run this first
    python scripts/train_model.py                   # Phase 8, full run

Refuses to start a full run until the baseline (Phase 4) exists, so the
experimental order in the brief cannot be reversed by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import plot_training_curves, run_training  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, set_seed  # noqa: E402

LOG = get_logger("train_model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Phase 9: tiny subset, 2 steps, verify the pipeline")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Cap the number of training examples (debugging)")
    parser.add_argument("--skip-baseline-check", action="store_true",
                        help="Override the baseline-first guard (not recommended)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    if not args.sanity_check and not args.skip_baseline_check:
        baseline_csv = resolve_path(cfg["baseline"]["results_csv"])
        if not baseline_csv.exists():
            raise SystemExit(
                "Baseline results not found at "
                f"{baseline_csv}.\n"
                "The experiment requires the BASE model to be measured before "
                "fine-tuning. Run:\n"
                "    python scripts/run_baseline.py\n"
                "(or pass --skip-baseline-check to override)"
            )

    summary = run_training(cfg, sanity=args.sanity_check, train_limit=args.train_limit)

    if args.sanity_check:
        print("\n" + "=" * 60)
        print("SANITY CHECK PASSED")
        print("=" * 60)
        print(f"Examples used     : {summary['n_train_examples']}")
        print(f"Steps completed   : {summary['global_step']}")
        print(f"Loss (finite)     : {summary['training_loss']:.4f}")
        print(f"Trainable params  : {summary['trainable_params']:,} "
              f"({summary['trainable_pct']}%)")
        print(f"Adapter saved to  : {summary['adapter_dir']}")
        print("\nNow run the full training:  python scripts/train_model.py")
        print("=" * 60)
        return

    curve = plot_training_curves(summary["log_history"],
                                 "results/training/training_curves.png")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final training loss : {summary['training_loss']:.4f}")
    print(f"Steps               : {summary['global_step']}")
    print(f"Trainable params    : {summary['trainable_params']:,} "
          f"({summary['trainable_pct']}%)")
    print(f"Adapter             : {summary['adapter_dir']}")
    if curve:
        print(f"Curves              : {curve}")
    print("\nNext: python scripts/evaluate_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
