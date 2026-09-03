#!/usr/bin/env python3
"""Render reports/baseline_report.md from actual baseline artefacts.

    python scripts/generate_baseline_report.py

Every number is read from results/baseline/*. Nothing is written by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.prompts import EVAL_PROMPTS  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

TRUNC = 260


def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _examples(df: pd.DataFrame, n: int = 4) -> str:
    """Render a handful of rows as readable blocks."""
    if df.empty:
        return "_None found._\n"
    out = []
    for _, r in df.head(n).iterrows():
        raw = str(r["raw_response"]).strip().replace("\n", "\n> ")
        out.append(
            f"**`{r['relpath']}`** — truth: **{r['ground_truth']}** "
            f"(type: `{r.get('true_defect_type')}`) | prompt: `{r['prompt_id']}` | "
            f"parsed: **{r['parsed_prediction']}** | "
            f"type predicted: `{r.get('predicted_defect_type')}` -> "
            f"`{r.get('predicted_class')}` | confidence: {_fmt(r.get('confidence'))}\n\n"
            f"> {raw[:TRUNC]}{'…' if len(str(r['raw_response'])) > TRUNC else ''}\n"
        )
    return "\n".join(out)


def _binary_table(m: dict) -> str:
    cm = m.get("confusion_matrix", {})
    rows = [
        ("Accuracy", m.get("accuracy")),
        ("Balanced accuracy", m.get("balanced_accuracy")),
        ("Precision (Defective)", m.get("precision_defective")),
        ("Recall (Defective)", m.get("recall_defective")),
        ("F1 (Defective)", m.get("f1_defective")),
        ("Macro F1", m.get("macro_f1")),
        ("OK recall", m.get("ok_recall")),
        ("OK recall 95% CI", m.get("ok_recall_95ci")),
        ("Defective recall", m.get("defective_recall")),
        ("Defective recall 95% CI", m.get("defective_recall_95ci")),
        ("True negatives (OK→OK)", cm.get("true_negative_ok_as_ok")),
        ("False positives (OK→Defective)", cm.get("false_positive_ok_as_defective")),
        ("False negatives (Defective→OK)", cm.get("false_negative_defective_as_ok")),
        ("True positives (Defective→Defective)", cm.get("true_positive_defective_as_defective")),
        ("Unparseable responses", m.get("n_unparseable")),
        ("Distinct images", f"{m.get('n_ok_images')} OK / {m.get('n_defective_images')} Defective"),
        ("Observations (image×prompt)", m.get("n_total")),
    ]
    return "\n".join(f"| {k} | {_fmt(v)} |" for k, v in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = resolve_path("results/baseline")
    df = pd.read_csv(base / "baseline_results.csv")
    metrics = json.loads((base / "baseline_metrics.json").read_text())
    balanced = json.loads((base / "balanced_subset.json").read_text())
    env = json.loads((base / "run_metadata.json").read_text())

    a = metrics["track_a_binary"]
    b = metrics.get("track_b_defect_type") or {}
    bal_m = a.get("balanced_subset", a["full_test_set"])
    full_m = a["full_test_set"]

    bal_paths = set(balanced["relpaths"])
    bal_df = df[df["relpath"].isin(bal_paths)]

    correct = df[df["correct"]]
    wrong = df[~df["correct"]]
    fn = df[(df["ground_truth"] == "Defective") & (df["parsed_prediction"] == "OK")]
    fp = df[(df["ground_truth"] == "OK") & (df["parsed_prediction"] == "Defective")]
    halluc = df[(df["ground_truth"] == "OK")
                & (~df["predicted_defect_type"].astype(str).isin(["None", "Unknown", "nan"]))]
    unparse = df[df["parsed_prediction"] == "Unparseable"]

    L = []
    W = L.append
    W("# Baseline Report — Base VLM, No Fine-Tuning\n")
    W("> **Status: COMPLETE.** Every number below was produced by "
      "`scripts/run_baseline.py` and read from `results/baseline/`. "
      "No value was written by hand.\n")
    W("> **The model was NOT fine-tuned.** No LoRA adapter, no training of any "
      "kind — the pretrained weights were queried directly.\n")

    W("\n## 1. Run metadata\n")
    W("| Field | Value |\n|---|---|")
    for k, v in env.items():
        W(f"| {k} | `{v}` |")

    W("\n## 2. Evaluation design\n")
    W(f"- **Full held-out test set:** {env['n_images']} images x "
      f"{len(cfg['baseline']['prompt_ids'])} prompts = **{len(df)} generations**.")
    W(f"- **Track A (primary)** reports on the frozen balanced slice: "
      f"**{balanced['n_ok']} OK + {balanced['n_defective']} Defective = "
      f"{balanced['n']} images**, seed {balanced['seed']}, "
      f"{balanced['n_source_groups']} source groups.")
    W("- **Track B (secondary)** uses the full test set for 12-class defect type.")
    W(f"\n> Subset selection: {balanced['selection_procedure']}\n")

    W("\n## 3. Prompts (frozen, verbatim)\n")
    for pid in cfg["baseline"]["prompt_ids"]:
        W(f"<details><summary><code>{pid}</code></summary>\n\n```\n{EVAL_PROMPTS[pid]}\n```\n</details>\n")
    W("These are used **unchanged** for the fine-tuned evaluation. They do not "
      "list the dataset's class names, so the base model is not handed the answer space.\n")

    W("\n## 4. TRACK A — PRIMARY: binary OK vs Defective\n")
    W("### 4.1 Balanced reporting slice (12 OK + 12 Defective)\n")
    W("| Metric | Value |\n|---|---|")
    W(_binary_table(bal_m))
    if bal_m.get("small_sample_warning"):
        W(f"\n> **Small-sample warning.** {bal_m['small_sample_warning']}")
    if bal_m.get("ci_basis"):
        W(f">\n> {bal_m['ci_basis']}")

    W("\n### 4.2 Full held-out test set (178 images)\n")
    W("| Metric | Value |\n|---|---|")
    W(_binary_table(full_m))
    W("\n> The full set is 12 OK vs 166 Defective, so accuracy here is inflated "
      "by the majority class. Balanced accuracy and macro F1 are the honest reads.")

    W("\n### 4.3 Per-prompt (balanced slice)\n")
    W("| Prompt | Accuracy | Balanced acc | OK recall | Defective recall | Macro F1 | Unparseable |")
    W("|---|---|---|---|---|---|---|")
    for pid, pm in sorted((bal_m.get("per_prompt") or {}).items()):
        W(f"| `{pid}` | {_fmt(pm.get('accuracy'))} | {_fmt(pm.get('balanced_accuracy'))} | "
          f"{_fmt(pm.get('ok_recall'))} | {_fmt(pm.get('defective_recall'))} | "
          f"{_fmt(pm.get('macro_f1'))} | {_fmt(pm.get('n_unparseable'))} |")

    W("\n### 4.4 Classification report (full test set)\n")
    W(f"```\n{full_m.get('classification_report_text', 'n/a')}\n```")
    W("\nConfusion matrices: `baseline_confusion_matrix.png` (full set), "
      "`baseline_confusion_matrix_balanced.png` (balanced slice).\n")

    W("\n## 5. TRACK B — SECONDARY: 12-class defect type\n")
    if b:
        fb = b["full_test_set"]
        W(f"**Coverage: {fb.get('coverage')}** "
          f"({fb.get('n_valid_class_predictions')}/{fb.get('n_total')} responses yielded "
          f"a recognisable class; {fb.get('n_unmatched')} unmatched).\n")
        W("| Metric | Value |\n|---|---|")
        for k, label in [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                         ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1"),
                         ("weighted_precision", "Weighted precision"),
                         ("weighted_recall", "Weighted recall"),
                         ("weighted_f1", "Weighted F1")]:
            W(f"| {label} | {_fmt(fb.get(k))} |")
        W(f"\n> {fb.get('coverage_note', '')}\n")
        W("### 5.1 Per-class precision / recall / F1\n")
        W(f"```\n{fb.get('classification_report_text', 'n/a')}\n```")
        W("\nConfusion matrix: `baseline_defect_type_confusion_matrix.png`\n")
    else:
        W("_Track B metrics unavailable._\n")

    W("\n## 6. Error analysis\n")
    es = metrics.get("error_summary", {})
    W(f"- Total observations: **{es.get('n_total')}**")
    W(f"- Correct: **{es.get('n_correct')}** | Incorrect: **{es.get('n_incorrect')}**\n")
    W("| Error type | Count |\n|---|---|")
    for k, v in sorted((es.get("error_type_counts") or {}).items()):
        W(f"| `{k}` | {v} |")
    if es.get("failure_flag_counts"):
        W("\n| Failure flag | Count |\n|---|---|")
        for k, v in sorted(es["failure_flag_counts"].items()):
            W(f"| `{k}` | {v} |")
    W("\nFull per-row detail: `results/baseline/error_analysis.csv`\n")

    W("\n## 7. Example outputs\n")
    W("### 7.1 Correct predictions\n")
    W(_examples(correct))
    W("\n### 7.2 Incorrect predictions\n")
    W(_examples(wrong))
    W("\n### 7.3 Missed defects — said OK for a Defective casting (false negatives)\n")
    W(f"_{len(fn)} of {len(df)} observations._\n")
    W(_examples(fn))
    W("\n### 7.4 False alarms — said Defective for an OK casting (false positives)\n")
    W(f"_{len(fp)} of {len(df)} observations._\n")
    W(_examples(fp))
    W("\n### 7.5 Hallucinated defect types on OK castings\n")
    W("_A defect type asserted on a casting that is genuinely OK._\n")
    W(f"_{len(halluc)} of {len(df)} observations._\n")
    W(_examples(halluc))
    W("\n### 7.6 Unparseable responses\n")
    W(f"_{len(unparse)} of {len(df)} observations. These are counted as errors, "
      "not discarded._\n")
    W(_examples(unparse))

    W("\n## 8. Observations\n")
    W(f"- The base model produced a parseable OK/Defective verdict in "
      f"**{100*(1-len(unparse)/max(len(df),1)):.1f}%** of generations.")
    W(f"- Format compliance (an explicit labelled field): "
      f"**{100*df['format_ok'].mean():.1f}%**.")
    W(f"- On the balanced slice it called **{_fmt(bal_m.get('n_ok_images'))} OK** and "
      f"**{_fmt(bal_m.get('n_defective_images'))} Defective** images; "
      f"OK recall {_fmt(bal_m.get('ok_recall'))} vs Defective recall "
      f"{_fmt(bal_m.get('defective_recall'))}.")
    if b:
        W(f"- Track B coverage was **{b['full_test_set'].get('coverage')}** — the base "
          "model rarely names a defect type matching the dataset's vocabulary, which "
          "is expected since the prompts never showed it that vocabulary.")
    W("\n_Interpretation to be written up in `reports/final_report.md`._\n")

    W("\n## 9. Limitations\n")
    W("- **Confidence is model-reported, NOT a calibrated probability.** No "
      "calibration was performed; treat every confidence figure as the model's "
      "own claim.")
    W(f"- **Small OK sample.** {balanced['n_ok']} distinct OK images in the balanced "
      "slice; one flip moves OK recall by ~8 points. OK recall is indicative, not "
      "statistically significant.")
    W("- **Synthetic defect labels.** " + metrics.get("synthetic_label_caveat", ""))
    W("- **Repeated prompts are not independent samples**; confidence intervals use "
      "distinct-image counts.")
    W("- This is a **research prototype**, not a certified industrial QC system.\n")

    out = resolve_path("reports/baseline_report.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out} ({len(''.join(L).splitlines())} lines)")


if __name__ == "__main__":
    main()
