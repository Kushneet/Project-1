#!/usr/bin/env python3
"""Phase 11 — compare the base model against the fine-tuned model.

    python scripts/compare_models.py

Reads the two results CSVs (same images, same prompts) and produces the
quantitative table, the qualitative example table and the comparison plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import compute_metrics  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, write_json  # noqa: E402

LOG = get_logger("compare")

METRIC_KEYS = [
    ("accuracy", "Accuracy"),
    ("precision_defective", "Precision (Defective)"),
    ("recall_defective", "Recall (Defective)"),
    ("f1_defective", "F1 (Defective)"),
    ("macro_f1", "Macro F1"),
]


def load_pair(base_csv: Path, ft_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both result sets and assert they cover the same evaluation."""
    for p, name in ((base_csv, "baseline"), (ft_csv, "fine-tuned")):
        if not p.exists():
            raise SystemExit(f"Missing {name} results at {p}. Run the {name} step first.")
    base, ft = pd.read_csv(base_csv), pd.read_csv(ft_csv)

    key = ["relpath", "prompt_id"]
    base_keys = set(map(tuple, base[key].values))
    ft_keys = set(map(tuple, ft[key].values))
    if base_keys != ft_keys:
        only_b, only_f = len(base_keys - ft_keys), len(ft_keys - base_keys)
        raise SystemExit(
            "Baseline and fine-tuned runs do NOT cover the same (image, prompt) "
            f"pairs: {only_b} only in baseline, {only_f} only in fine-tuned. "
            "The comparison would be invalid. Re-run evaluation with the frozen subset."
        )
    return base, ft


def comparison_table(base: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Build the headline metric table, overall and per prompt."""
    rows = []

    def _add(scope: str, b: pd.DataFrame, f: pd.DataFrame) -> None:
        mb, mf = compute_metrics(b), compute_metrics(f)
        for key, label in METRIC_KEYS:
            bv, fv = mb.get(key), mf.get(key)
            rows.append({
                "scope": scope, "metric": label,
                "base_model": bv, "finetuned_model": fv,
                "delta": None if bv is None or fv is None else round(fv - bv, 4),
            })
        cmb, cmf = mb.get("confusion_matrix", {}), mf.get("confusion_matrix", {})
        for key, label in (("false_positive_ok_as_defective", "False Positives"),
                           ("false_negative_defective_as_ok", "False Negatives")):
            bv, fv = cmb.get(key), cmf.get(key)
            rows.append({"scope": scope, "metric": label, "base_model": bv,
                         "finetuned_model": fv,
                         "delta": None if bv is None or fv is None else fv - bv})
        rows.append({"scope": scope, "metric": "Unparseable outputs",
                     "base_model": mb.get("n_unparseable"),
                     "finetuned_model": mf.get("n_unparseable"),
                     "delta": (mf.get("n_unparseable", 0) - mb.get("n_unparseable", 0))})

    _add("overall", base, ft)
    for pid in sorted(set(base["prompt_id"])):
        _add(pid, base[base["prompt_id"] == pid], ft[ft["prompt_id"] == pid])
    return pd.DataFrame(rows)


def qualitative_table(base: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Join per-image predictions and label the four agreement quadrants."""
    cols = ["relpath", "prompt_id", "ground_truth", "parsed_prediction",
            "predicted_defect_type", "confidence", "raw_response"]
    b = base[cols].rename(columns={
        "parsed_prediction": "base_prediction",
        "predicted_defect_type": "base_defect_type",
        "confidence": "base_confidence", "raw_response": "base_response"})
    f = ft[cols].rename(columns={
        "parsed_prediction": "finetuned_prediction",
        "predicted_defect_type": "finetuned_defect_type",
        "confidence": "finetuned_confidence", "raw_response": "finetuned_response"})
    merged = b.merge(f, on=["relpath", "prompt_id", "ground_truth"], how="inner")

    merged["base_correct"] = merged["ground_truth"] == merged["base_prediction"]
    merged["finetuned_correct"] = merged["ground_truth"] == merged["finetuned_prediction"]
    merged["quadrant"] = [
        "both_correct" if bc and fc
        else "both_wrong" if not bc and not fc
        else "fixed_by_finetuning" if not bc and fc
        else "broken_by_finetuning"
        for bc, fc in zip(merged["base_correct"], merged["finetuned_correct"])
    ]
    return merged


def plot_comparison(table: pd.DataFrame, out_path: Path) -> Path:
    """Grouped bar chart of the overall headline metrics."""
    overall = table[(table["scope"] == "overall")
                    & (table["metric"].isin([lbl for _, lbl in METRIC_KEYS]))]
    labels = overall["metric"].tolist()
    base_vals = [v if pd.notna(v) else 0 for v in overall["base_model"]]
    ft_vals = [v if pd.notna(v) else 0 for v in overall["finetuned_model"]]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base model", color="#9AA6B2")
    ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-tuned", color="#4C72B0")
    ax.set_xticks(list(x), labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Base vs fine-tuned — same images, same prompts")
    for i, (b, f) in enumerate(zip(base_vals, ft_vals)):
        ax.text(i - width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg["evaluation"]["comparison_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    base, ft = load_pair(resolve_path(cfg["baseline"]["results_csv"]),
                         resolve_path(cfg["evaluation"]["results_csv"]))

    table = comparison_table(base, ft)
    table.to_csv(out_dir / "comparison.csv", index=False)

    qual = qualitative_table(base, ft)
    qual.to_csv(out_dir / "qualitative_comparison.csv", index=False)

    plot_comparison(table, out_dir / "comparison_plot.png")
    counts = qual["quadrant"].value_counts().to_dict()
    write_json({"quadrant_counts": counts,
                "n_compared_rows": int(len(qual))},
               out_dir / "comparison_summary.json")

    print("\n" + "=" * 72)
    print("BASE vs FINE-TUNED".center(72))
    print("=" * 72)
    overall = table[table["scope"] == "overall"]
    print(f"{'Metric':<26}{'Base':>12}{'Fine-tuned':>14}{'Delta':>12}")
    print("-" * 72)
    for _, r in overall.iterrows():
        d = "" if r["delta"] is None or pd.isna(r["delta"]) else f"{r['delta']:+.4f}"
        print(f"{r['metric']:<26}{str(r['base_model']):>12}"
              f"{str(r['finetuned_model']):>14}{d:>12}")
    print("-" * 72)
    print("Per-image agreement:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24}{v:>6}")
    print("=" * 72)
    print(f"Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
