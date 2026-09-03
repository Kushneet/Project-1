"""Tests for metrics, error categorisation and confusion-matrix plotting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (  # noqa: E402
    add_error_columns,
    compute_metrics,
    error_analysis,
    error_summary,
    error_type,
    plot_confusion_matrix,
)


def _results(preds, truths, **extra) -> pd.DataFrame:
    df = pd.DataFrame({"ground_truth": truths, "parsed_prediction": preds})
    for k, v in extra.items():
        df[k] = v
    return df


class TestErrorType:
    def test_correct(self):
        assert error_type("OK", "OK") == "correct"

    def test_false_negative(self):
        assert error_type("Defective", "OK") == "false_negative_missed_defect"

    def test_false_positive(self):
        assert error_type("OK", "Defective") == "false_positive_false_alarm"

    def test_unparseable(self):
        assert error_type("OK", "Unparseable") == "unparseable_output"


class TestComputeMetrics:
    def test_perfect_predictions(self):
        m = compute_metrics(_results(["OK", "Defective"] * 5, ["OK", "Defective"] * 5))
        assert m["accuracy"] == 1.0
        assert m["f1_defective"] == 1.0
        cm = m["confusion_matrix"]
        assert cm["false_positive_ok_as_defective"] == 0
        assert cm["false_negative_defective_as_ok"] == 0

    def test_all_wrong(self):
        m = compute_metrics(_results(["Defective"] * 4 + ["OK"] * 4,
                                     ["OK"] * 4 + ["Defective"] * 4))
        assert m["accuracy"] == 0.0

    def test_confusion_matrix_orientation(self):
        # 3 defective predicted OK -> false negatives; 1 OK predicted Defective.
        df = _results(
            preds=["OK", "OK", "OK", "Defective", "OK"],
            truths=["Defective", "Defective", "Defective", "OK", "OK"],
        )
        cm = compute_metrics(df)["confusion_matrix"]
        assert cm["false_negative_defective_as_ok"] == 3
        assert cm["false_positive_ok_as_defective"] == 1
        assert cm["true_negative_ok_as_ok"] == 1

    def test_always_defective_gets_perfect_recall_poor_precision(self):
        # The degenerate "everything is defective" baseline must be exposed.
        df = _results(["Defective"] * 10, ["Defective"] * 5 + ["OK"] * 5)
        m = compute_metrics(df)
        assert m["recall_defective"] == 1.0
        assert m["precision_defective"] == 0.5
        assert m["accuracy"] == 0.5

    def test_unparseable_counted_as_error_by_default(self):
        df = _results(["Unparseable"] * 4, ["OK", "Defective"] * 2)
        m = compute_metrics(df)
        assert m["n_unparseable"] == 4
        assert m["accuracy"] == 0.0, "unparseable output must not be scored as correct"

    def test_unparseable_can_be_excluded_explicitly(self):
        df = _results(["OK", "Unparseable"], ["OK", "Defective"])
        m = compute_metrics(df, treat_unparseable_as_error=False)
        assert m["n_scored"] == 1
        assert m["accuracy"] == 1.0

    def test_unparseable_rate_reported(self):
        df = _results(["OK", "Unparseable", "OK", "OK"], ["OK"] * 4)
        assert compute_metrics(df)["unparseable_rate"] == 0.25

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            compute_metrics(pd.DataFrame({"foo": [1]}))

    def test_metrics_are_json_serialisable(self):
        import json

        m = compute_metrics(_results(["OK", "Defective"], ["OK", "Defective"]))
        json.dumps(m, default=str)


class TestErrorAnalysis:
    def test_only_incorrect_rows_returned(self):
        df = _results(["OK", "Defective", "OK"], ["OK", "OK", "Defective"])
        errs = error_analysis(df)
        assert len(errs) == 2

    def test_hallucinated_defect_type_flagged(self):
        df = _results(["Defective"], ["OK"], predicted_defect_type=["crack"],
                      confidence=[95.0], format_ok=[True])
        errs = error_analysis(df)
        assert "hallucinated_defect_type" in errs.iloc[0]["failure_flags"]
        assert "confidently_wrong" in errs.iloc[0]["failure_flags"]

    def test_no_errors_returns_empty(self):
        df = _results(["OK", "Defective"], ["OK", "Defective"])
        assert error_analysis(df).empty

    def test_summary_counts(self):
        df = _results(["OK", "Defective", "OK"], ["OK", "OK", "Defective"])
        s = error_summary(df)
        assert s["n_correct"] == 1
        assert s["n_incorrect"] == 2
        assert s["error_type_counts"]["false_positive_false_alarm"] == 1

    def test_add_error_columns(self):
        out = add_error_columns(_results(["OK"], ["Defective"]))
        assert out.iloc[0]["correct"] is False or not out.iloc[0]["correct"]
        assert out.iloc[0]["error_type"] == "false_negative_missed_defect"


class TestPlot:
    def test_confusion_matrix_png_written(self, tmp_path):
        m = compute_metrics(_results(["OK", "Defective"], ["OK", "Defective"]))
        p = plot_confusion_matrix(m, tmp_path / "cm.png", "test")
        assert p.exists() and p.stat().st_size > 0
