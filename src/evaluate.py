"""Metrics, error categorisation and confusion-matrix plotting.

Used identically by the baseline (Phase 4/5) and the fine-tuned evaluation
(Phase 10) so the two are directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .prompts import BINARY_DEFECTIVE, BINARY_OK, UNPARSEABLE
from .utils import get_logger, resolve_path, write_json

LOG = get_logger("evaluate")

LABELS = [BINARY_OK, BINARY_DEFECTIVE]
POSITIVE = BINARY_DEFECTIVE  # "defective" is the positive class


def error_type(ground_truth: str, prediction: str) -> str:
    """Categorise one prediction against its ground truth."""
    if prediction == UNPARSEABLE:
        return "unparseable_output"
    if ground_truth == prediction:
        return "correct"
    if ground_truth == BINARY_DEFECTIVE and prediction == BINARY_OK:
        return "false_negative_missed_defect"
    if ground_truth == BINARY_OK and prediction == BINARY_DEFECTIVE:
        return "false_positive_false_alarm"
    return "other"


def compute_metrics(df: pd.DataFrame, treat_unparseable_as_error: bool = True) -> dict[str, Any]:
    """Compute binary OK/Defective metrics from a results DataFrame.

    Unparseable outputs are, by default, counted as errors rather than
    dropped — silently discarding them would flatter a model that fails to
    follow the requested format. Both variants are reported.
    """
    required = {"ground_truth", "parsed_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Results frame missing columns: {sorted(missing)}")

    n_total = int(len(df))
    n_unparseable = int((df["parsed_prediction"] == UNPARSEABLE).sum())

    work = df.copy()
    if treat_unparseable_as_error:
        # Map an unparseable output to the wrong label so it counts against the model.
        flip = {BINARY_OK: BINARY_DEFECTIVE, BINARY_DEFECTIVE: BINARY_OK}
        mask = work["parsed_prediction"] == UNPARSEABLE
        work.loc[mask, "parsed_prediction"] = work.loc[mask, "ground_truth"].map(flip)
        scored = work
    else:
        scored = work[work["parsed_prediction"] != UNPARSEABLE]

    if scored.empty:
        return {
            "n_total": n_total,
            "n_unparseable": n_unparseable,
            "note": "no parseable predictions; metrics undefined",
        }

    y_true = scored["ground_truth"].tolist()
    y_pred = scored["parsed_prediction"].tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, pos_label=POSITIVE, average="binary",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "n_total": n_total,
        "n_scored": int(len(scored)),
        "n_unparseable": n_unparseable,
        "unparseable_rate": round(n_unparseable / n_total, 4) if n_total else 0.0,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_defective": round(float(precision), 4),
        "recall_defective": round(float(recall), 4),
        "f1_defective": round(float(f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist(),
            "true_negative_ok_as_ok": int(tn),
            "false_positive_ok_as_defective": int(fp),
            "false_negative_defective_as_ok": int(fn),
            "true_positive_defective_as_defective": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
        "positive_class": POSITIVE,
        "unparseable_treated_as_error": treat_unparseable_as_error,
    }


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``correct`` and ``error_type`` columns to a results frame."""
    out = df.copy()
    out["correct"] = out["ground_truth"] == out["parsed_prediction"]
    out["error_type"] = [
        error_type(gt, pred)
        for gt, pred in zip(out["ground_truth"], out["parsed_prediction"])
    ]
    return out


def error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the incorrect rows, annotated for qualitative review."""
    annotated = add_error_columns(df)
    errors = annotated[~annotated["correct"]].copy()
    if errors.empty:
        return errors

    def _flags(row: pd.Series) -> str:
        notes = []
        if not row.get("format_ok", True):
            notes.append("poor_formatting")
        dt = str(row.get("predicted_defect_type", "") or "")
        # A defect type asserted on a genuinely OK casting is a hallucination.
        if row["ground_truth"] == BINARY_OK and dt not in {"None", "Unknown", ""}:
            notes.append("hallucinated_defect_type")
        conf = row.get("confidence")
        if pd.notna(conf) and conf is not None and float(conf) >= 90:
            notes.append("confidently_wrong")
        if pd.notna(conf) and conf is not None and float(conf) <= 50:
            notes.append("low_confidence")
        return ";".join(notes) if notes else "none"

    errors["failure_flags"] = errors.apply(_flags, axis=1)
    return errors


def error_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate counts of each error category."""
    annotated = add_error_columns(df)
    summary = {
        "n_total": int(len(annotated)),
        "n_correct": int(annotated["correct"].sum()),
        "n_incorrect": int((~annotated["correct"]).sum()),
        "error_type_counts": annotated["error_type"].value_counts().to_dict(),
    }
    errs = error_analysis(df)
    if not errs.empty and "failure_flags" in errs.columns:
        flags: dict[str, int] = {}
        for row in errs["failure_flags"]:
            for f in str(row).split(";"):
                if f and f != "none":
                    flags[f] = flags.get(f, 0) + 1
        summary["failure_flag_counts"] = flags
    return summary


def plot_confusion_matrix(metrics: dict[str, Any], out_path: str | Path, title: str) -> Path:
    """Render a labelled confusion matrix to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = metrics["confusion_matrix"]["matrix"]
    labels = metrics["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    total = sum(sum(r) for r in cm) or 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i][j]}\n({100*cm[i][j]/total:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i][j] > max(max(r) for r in cm) / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def evaluate_results_file(
    results_csv: str | Path,
    out_dir: str | Path,
    tag: str,
    balanced_subset: str | Path | None = None,
) -> dict[str, Any]:
    """Score a results CSV for both tracks, writing metrics, errors and plots.

    Track A (primary) is binary OK/Defective, reported on the balanced subset
    when one is supplied and on the full held-out test set either way.
    Track B (secondary) is 12-class defect type on the full test set.
    """
    df = pd.read_csv(resolve_path(results_csv))
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    balanced = _load_relpath_filter(balanced_subset)

    # ---- TRACK A (primary): binary OK vs Defective ----
    track_a: dict[str, Any] = {
        "full_test_set": binary_track_metrics(df),
        "per_prompt": {str(pid): binary_track_metrics(sub)
                       for pid, sub in df.groupby("prompt_id")},
    }
    if balanced is not None:
        sub = df[df["relpath"].isin(balanced)]
        track_a["balanced_subset"] = binary_track_metrics(sub)
        track_a["balanced_subset"]["per_prompt"] = {
            str(pid): binary_track_metrics(g) for pid, g in sub.groupby("prompt_id")}
        track_a["primary_view"] = "balanced_subset"
    else:
        track_a["primary_view"] = "full_test_set"

    # ---- TRACK B (secondary): 12-class defect type ----
    track_b: dict[str, Any] = {}
    if "predicted_class" in df.columns and "true_defect_type" in df.columns:
        track_b = {
            "full_test_set": compute_multiclass_metrics(df),
            "per_prompt": {str(pid): compute_multiclass_metrics(sub)
                           for pid, sub in df.groupby("prompt_id")},
        }
        plot_multiclass_confusion(
            track_b["full_test_set"],
            out / f"{tag}_defect_type_confusion_matrix.png",
            f"{tag}: 12-class defect type (synthetic labels)")

    payload = {
        "tag": tag,
        "track_a_binary": track_a,
        "track_b_defect_type": track_b,
        "error_summary": error_summary(df),
        "synthetic_label_caveat": (
            "The 12 defect classes were generated programmatically by painting "
            "defects onto real OK castings. Track B measures recognition of "
            "synthetic defect textures, NOT real industrial defect recognition."
        ),
        # kept for backward compatibility with earlier tooling
        "overall": track_a["full_test_set"],
    }
    write_json(payload, out / f"{tag}_metrics.json")

    errs = error_analysis(df)
    if not errs.empty:
        errs.to_csv(out / "error_analysis.csv", index=False)
    plot_confusion_matrix(track_a["full_test_set"],
                          out / f"{tag}_confusion_matrix.png",
                          f"{tag}: OK vs Defective (full held-out test set)")
    if balanced is not None:
        plot_confusion_matrix(track_a["balanced_subset"],
                              out / f"{tag}_confusion_matrix_balanced.png",
                              f"{tag}: OK vs Defective (balanced subset)")
    LOG.info("Wrote Track A + Track B metrics for %s to %s", tag, out)
    return payload


def _load_relpath_filter(subset_path: str | Path | None) -> set[str] | None:
    """Read a frozen subset file and return its relpaths, if it exists."""
    import json

    if subset_path is None:
        return None
    p = resolve_path(subset_path)
    if not p.exists():
        LOG.warning("Balanced subset file %s not found; reporting on the full set only", p)
        return None
    return set(json.loads(p.read_text(encoding="utf-8"))["relpaths"])


# ---------------------------------------------------------------------------
# TRACK B — 12-class defect-type evaluation
# ---------------------------------------------------------------------------

def compute_multiclass_metrics(
    df: pd.DataFrame,
    classes: list[str] | None = None,
    scored_only: bool = True,
) -> dict[str, Any]:
    """Score 12-class defect-type prediction.

    ``scored_only=True`` restricts metrics to rows where the model produced a
    recognisable class (the brief's "where the model can produce a valid class
    prediction"). Coverage is always reported, because metrics computed on a
    self-selected subset are optimistically biased and that must be visible.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support

    from .prompts import DEFECT_CLASSES, UNMATCHED

    labels = classes or DEFECT_CLASSES
    required = {"true_defect_type", "predicted_class"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Results frame missing columns: {sorted(missing)}")

    n_total = int(len(df))
    valid = df[df["predicted_class"] != UNMATCHED]
    n_valid = int(len(valid))

    payload: dict[str, Any] = {
        "n_total": n_total,
        "n_valid_class_predictions": n_valid,
        "coverage": round(n_valid / n_total, 4) if n_total else 0.0,
        "n_unmatched": n_total - n_valid,
        "scored_only": scored_only,
        "classes": labels,
        "coverage_note": (
            "Metrics below are computed only on rows where the model produced a "
            "recognisable class. This subset is self-selected, so the scores are "
            "optimistically biased; read them together with coverage."
        ),
    }

    scored = valid if scored_only else df
    if scored.empty:
        payload["note"] = "no valid class predictions; Track B metrics undefined"
        return payload

    y_true = scored["true_defect_type"].astype(str).tolist()
    y_pred = scored["predicted_class"].astype(str).tolist()
    all_labels = labels if scored_only else labels + [UNMATCHED]

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)

    payload.update({
        "n_scored": int(len(scored)),
        "accuracy": round(float((pd.Series(y_true) == pd.Series(y_pred)).mean()), 4),
        "macro_precision": round(float(macro[0]), 4),
        "macro_recall": round(float(macro[1]), 4),
        "macro_f1": round(float(macro[2]), 4),
        "weighted_precision": round(float(weighted[0]), 4),
        "weighted_recall": round(float(weighted[1]), 4),
        "weighted_f1": round(float(weighted[2]), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=all_labels, zero_division=0),
        "confusion_matrix": {
            "labels": all_labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        },
    })
    return payload


def plot_multiclass_confusion(metrics: dict[str, Any], out_path: str | Path,
                              title: str) -> Path | None:
    """Render the 12-class confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_block = metrics.get("confusion_matrix")
    if not cm_block:
        return None
    cm = cm_block["matrix"]
    labels = cm_block["labels"]

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted defect type")
    ax.set_ylabel("True defect type")
    ax.set_title(title)
    peak = max((max(r) for r in cm), default=0) or 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i][j]:
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i][j] > peak / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def binary_track_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Track A metrics, adding explicit per-class recall and small-sample flags."""
    m = compute_metrics(df)
    cm = m.get("confusion_matrix", {})
    tn = cm.get("true_negative_ok_as_ok", 0)
    fp = cm.get("false_positive_ok_as_defective", 0)
    fn = cm.get("false_negative_defective_as_ok", 0)
    tp = cm.get("true_positive_defective_as_defective", 0)

    n_ok = tn + fp      # observations, i.e. (image, prompt) rows
    n_def = tp + fn
    m["n_ok_observations"] = int(n_ok)
    m["n_defective_observations"] = int(n_def)
    m["ok_recall"] = round(tn / n_ok, 4) if n_ok else None
    m["defective_recall"] = round(tp / n_def, 4) if n_def else None
    m["balanced_accuracy"] = (
        round((m["ok_recall"] + m["defective_recall"]) / 2, 4)
        if n_ok and n_def else None
    )

    # Each image is queried with several prompts, so rows are NOT independent
    # samples. Confidence intervals and the small-sample warning are therefore
    # based on the number of DISTINCT IMAGES, which is the conservative choice.
    if "relpath" in df.columns:
        uniq = df.drop_duplicates("relpath")
        n_ok_img = int((uniq["ground_truth"] == BINARY_OK).sum())
        n_def_img = int((uniq["ground_truth"] == BINARY_DEFECTIVE).sum())
        n_prompts = int(df.groupby("relpath").size().max()) if len(df) else 1
    else:
        n_ok_img, n_def_img, n_prompts = int(n_ok), int(n_def), 1

    m["n_ok_images"] = n_ok_img
    m["n_defective_images"] = n_def_img
    m["n_prompts_per_image"] = n_prompts
    m["ci_basis"] = (
        f"Wilson 95% intervals use {n_ok_img} distinct OK / {n_def_img} distinct "
        f"Defective images, not the {int(n_ok)}/{int(n_def)} (image, prompt) rows: "
        "repeated prompts on the same image are not independent samples."
    )
    m["ok_recall_95ci"] = (
        _wilson_interval(round((m["ok_recall"] or 0) * n_ok_img), n_ok_img)
        if n_ok_img else None
    )
    m["defective_recall_95ci"] = (
        _wilson_interval(round((m["defective_recall"] or 0) * n_def_img), n_def_img)
        if n_def_img else None
    )
    m["small_sample_warning"] = (
        f"OK class has only {n_ok_img} distinct images; one flip changes OK recall "
        f"by {100/n_ok_img:.1f} points. Treat OK recall as indicative, not "
        "statistically significant."
        if 0 < n_ok_img < 30 else None
    )
    return m


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score 95% confidence interval for a proportion."""
    if not n:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
