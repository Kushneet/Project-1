"""Tests for Track A (binary) and Track B (12-class) evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import build_balanced_subset  # noqa: E402
from src.evaluate import (  # noqa: E402
    _wilson_interval,
    binary_track_metrics,
    compute_multiclass_metrics,
)
from src.prompts import DEFECT_CLASSES, UNMATCHED, canonicalize_defect_type  # noqa: E402


class TestCanonicalize:
    @pytest.mark.parametrize("text,expected", [
        ("crack", "crack"),
        ("Crack", "crack"),
        ("cold shut", "cold_shut"),
        ("cold_shut", "cold_shut"),
        ("surface roughness", "surface_roughness"),
        ("mixed_defects", "mixed_defects"),
        ("blowhole", "porosity"),
        ("gas holes", "pinhole"),
        ("sink mark", "shrinkage"),
        ("burr along the parting line", "flash"),
        ("foreign material embedded", "inclusion"),
        ("linear mark from tooling", "scratch"),
    ])
    def test_synonyms(self, text, expected):
        assert canonicalize_defect_type(text) == expected

    def test_all_twelve_labels_round_trip(self):
        for cls in DEFECT_CLASSES:
            assert canonicalize_defect_type(cls) == cls

    def test_ok_prediction_maps_to_ok(self):
        assert canonicalize_defect_type(None, prediction="OK") == "ok"
        assert canonicalize_defect_type("anything", prediction="OK") == "ok"

    @pytest.mark.parametrize("text", ["Unknown", "", None, "n/a", "purple elephant"])
    def test_unrecognised_is_unmatched_not_guessed(self, text):
        assert canonicalize_defect_type(text) == UNMATCHED

    def test_longest_synonym_wins(self):
        # "cold shut" must beat any shorter incidental match
        assert canonicalize_defect_type("a cold shut seam") == "cold_shut"


class TestBinaryTrack:
    @staticmethod
    def _df(preds, truths):
        return pd.DataFrame({"ground_truth": truths, "parsed_prediction": preds})

    def test_per_class_recall_reported(self):
        m = binary_track_metrics(self._df(
            ["OK", "OK", "Defective", "Defective"],
            ["OK", "Defective", "Defective", "Defective"]))
        assert m["n_ok_observations"] == 1
        assert m["n_defective_observations"] == 3
        assert m["ok_recall"] == 1.0
        assert m["defective_recall"] == pytest.approx(2 / 3, abs=1e-3)

    def test_balanced_accuracy(self):
        m = binary_track_metrics(self._df(["OK", "Defective"], ["OK", "Defective"]))
        assert m["balanced_accuracy"] == 1.0

    def test_small_sample_warning_fires(self):
        m = binary_track_metrics(self._df(["OK"] * 12 + ["Defective"] * 12,
                                          ["OK"] * 12 + ["Defective"] * 12))
        assert m["small_sample_warning"] is not None
        assert "12" in m["small_sample_warning"]

    def test_repeated_prompts_do_not_inflate_the_sample(self):
        """3 prompts on 12 OK images is 12 samples, not 36."""
        df = pd.DataFrame({
            "ground_truth": ["OK"] * 36,
            "parsed_prediction": ["OK"] * 36,
            "relpath": [f"ok/{i}.jpg" for i in range(12)] * 3,
        })
        m = binary_track_metrics(df)
        assert m["n_ok_observations"] == 36
        assert m["n_ok_images"] == 12
        assert m["n_prompts_per_image"] == 3
        assert m["small_sample_warning"] is not None, \
            "warning must fire on 12 distinct images despite 36 rows"
        lo, _ = m["ok_recall_95ci"]
        assert lo < 0.8, "CI must be based on 12 images, not 36 rows"

    def test_no_warning_on_large_sample(self):
        m = binary_track_metrics(self._df(["OK"] * 40 + ["Defective"] * 40,
                                          ["OK"] * 40 + ["Defective"] * 40))
        assert m["small_sample_warning"] is None

    def test_confidence_interval_present_and_wide_when_small(self):
        m = binary_track_metrics(self._df(["OK"] * 12, ["OK"] * 12))
        lo, hi = m["ok_recall_95ci"]
        assert 0.0 <= lo <= hi <= 1.0
        assert lo < 0.9, "12/12 successes must still carry a wide lower bound"


class TestWilson:
    def test_perfect_score_not_claimed_as_certain(self):
        lo, hi = _wilson_interval(12, 12)
        assert hi == 1.0 and lo < 0.8

    def test_half(self):
        lo, hi = _wilson_interval(50, 100)
        assert lo < 0.5 < hi

    def test_zero_n(self):
        assert _wilson_interval(0, 0) is None


class TestMulticlassTrack:
    @staticmethod
    def _df(true, pred):
        return pd.DataFrame({"true_defect_type": true, "predicted_class": pred})

    def test_perfect(self):
        m = compute_multiclass_metrics(self._df(["crack", "ok"], ["crack", "ok"]))
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] > 0
        assert m["coverage"] == 1.0

    def test_coverage_reported_and_unmatched_excluded(self):
        m = compute_multiclass_metrics(
            self._df(["crack", "ok", "dent"], ["crack", UNMATCHED, "dent"]))
        assert m["n_unmatched"] == 1
        assert m["coverage"] == pytest.approx(2 / 3, abs=1e-3)
        assert m["n_scored"] == 2

    def test_coverage_bias_is_documented(self):
        m = compute_multiclass_metrics(self._df(["crack"], ["crack"]))
        assert "biased" in m["coverage_note"]

    def test_all_twelve_labels_in_confusion_matrix(self):
        m = compute_multiclass_metrics(self._df(["crack"], ["crack"]))
        assert m["confusion_matrix"]["labels"] == DEFECT_CLASSES
        assert len(m["confusion_matrix"]["matrix"]) == 12

    def test_no_valid_predictions_is_reported_not_crashed(self):
        m = compute_multiclass_metrics(self._df(["crack", "ok"], [UNMATCHED] * 2))
        assert m["coverage"] == 0.0
        assert "undefined" in m.get("note", "")

    def test_macro_and_weighted_both_present(self):
        m = compute_multiclass_metrics(
            self._df(["crack"] * 5 + ["dent"], ["crack"] * 5 + ["crack"]))
        for k in ("macro_precision", "macro_recall", "macro_f1",
                  "weighted_precision", "weighted_recall", "weighted_f1"):
            assert k in m

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            compute_multiclass_metrics(pd.DataFrame({"x": [1]}))


class TestBalancedSubset:
    @staticmethod
    def _examples(n_ok=12, n_def=166):
        ex = [{"relpath": f"ok/ok_{i:03d}.jpg", "image": f"/x/ok_{i}.jpg",
               "image_id": f"ok_{i}", "ground_truth": "OK", "defect_type": "ok",
               "group_key": f"g{i:03d}"} for i in range(n_ok)]
        types = ["crack", "dent", "flash", "porosity", "scratch"]
        ex += [{"relpath": f"{types[i%5]}/d_{i:03d}.jpg", "image": f"/x/d_{i}.jpg",
                "image_id": f"d_{i}", "ground_truth": "Defective",
                "defect_type": types[i % 5], "group_key": f"h{i:03d}"}
               for i in range(n_def)]
        return ex

    def test_is_balanced(self, tmp_path):
        p = build_balanced_subset(self._examples(), subset_path=tmp_path / "b.json")
        assert p["n_ok"] == p["n_defective"] == 12
        assert p["n"] == 24

    def test_takes_every_ok_image(self, tmp_path):
        p = build_balanced_subset(self._examples(), subset_path=tmp_path / "b.json")
        oks = [r for r in p["relpaths"] if r.startswith("ok/")]
        assert len(oks) == 12

    def test_spreads_across_defect_types(self, tmp_path):
        p = build_balanced_subset(self._examples(), subset_path=tmp_path / "b.json")
        types = {k for k in p["defect_type_counts"] if k != "ok"}
        assert len(types) >= 5, "defective side must span defect types, not one class"

    def test_records_provenance(self, tmp_path):
        p = build_balanced_subset(self._examples(), subset_path=tmp_path / "b.json")
        for k in ("seed", "selection_procedure", "source_groups", "n_source_groups"):
            assert k in p

    def test_is_frozen_and_reused(self, tmp_path):
        f = tmp_path / "b.json"
        a = build_balanced_subset(self._examples(), seed=1, subset_path=f)
        b = build_balanced_subset(self._examples(), seed=999, subset_path=f, reuse=True)
        assert a["relpaths"] == b["relpaths"], "frozen subset must not be resampled"

    def test_deterministic(self, tmp_path):
        a = build_balanced_subset(self._examples(), seed=7, subset_path=tmp_path/"a.json")
        b = build_balanced_subset(self._examples(), seed=7, subset_path=tmp_path/"b.json")
        assert a["relpaths"] == b["relpaths"]

    def test_subset_is_drawn_only_from_the_held_out_set(self, tmp_path):
        ex = self._examples()
        p = build_balanced_subset(ex, subset_path=tmp_path / "b.json")
        available = {e["relpath"] for e in ex}
        assert set(p["relpaths"]) <= available
