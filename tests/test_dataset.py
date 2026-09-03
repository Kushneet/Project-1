"""Tests for indexing, group derivation and leakage-safe splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (  # noqa: E402
    derive_group_key,
    drop_exact_duplicates,
    is_defective_label,
    make_splits,
    split_statistics,
    verify_no_leakage,
)


def _frame(n_groups: int = 40, per_group: int = 4) -> pd.DataFrame:
    """Synthetic index: each group has several images sharing one label."""
    rows = []
    for g in range(n_groups):
        label = "OK" if g % 4 == 0 else "Defective"
        dtype = "ok" if label == "OK" else ["crack", "porosity", "dent"][g % 3]
        for k in range(per_group):
            rows.append(
                {
                    "relpath": f"{dtype}/img_{g:04d}_{k}.jpeg",
                    "filename": f"img_{g:04d}_{k}.jpeg",
                    "class_label": dtype,
                    "defect_type": dtype,
                    "binary_label": label,
                    "is_defective": label == "Defective",
                    "group_key": f"{g:04d}",
                    "content_hash": f"h{g:04d}{k}",
                    "is_valid": True,
                }
            )
    return pd.DataFrame(rows)


class TestLabelMapping:
    @pytest.mark.parametrize("label", ["ok", "OK", "ok_front", "good", "no_defect"])
    def test_ok_labels(self, label):
        assert is_defective_label(label) is False

    @pytest.mark.parametrize(
        "label", ["crack", "porosity", "cold_shut", "mixed_defects", "def_front"]
    )
    def test_defective_labels(self, label):
        assert is_defective_label(label) is True

    def test_unknown_label_defaults_to_defective_not_ok(self):
        # Fail safe: an unrecognised folder must never be silently called OK.
        assert is_defective_label("some_new_folder") is True


class TestGroupKey:
    def test_numeric_id_recovered(self):
        assert derive_group_key("crack/cast_0123_7.jpeg", "cast_0123_7.jpeg") == "0123"

    def test_same_casting_different_class_shares_group(self):
        a = derive_group_key("ok/cast_0456.jpeg", "cast_0456.jpeg")
        b = derive_group_key("crack/cast_0456_hand_12.jpeg", "cast_0456_hand_12.jpeg")
        assert a == b == "0456"

    def test_non_numeric_falls_back_to_stem(self):
        assert derive_group_key("ok/casting.jpeg", "casting.jpeg") == "casting"


class TestSplitting:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError):
            make_splits(_frame(), 0.5, 0.3, 0.3)

    def test_all_images_assigned(self):
        out = make_splits(_frame(), seed=42)
        assert out["split"].notna().all()
        assert set(out["split"].unique()) <= {"train", "validation", "test"}

    def test_no_group_spans_splits(self):
        out = make_splits(_frame(), seed=42)
        assert out.groupby("group_key")["split"].nunique().max() == 1

    def test_verify_no_leakage_passes_on_clean_split(self):
        out = make_splits(_frame(), seed=42)
        assert verify_no_leakage(out)["leakage_free"] is True

    def test_verify_no_leakage_raises_on_planted_leak(self):
        out = make_splits(_frame(), seed=42)
        # Force one image of a train group into test.
        train_group = out[out["split"] == "train"]["group_key"].iloc[0]
        idx = out[out["group_key"] == train_group].index[0]
        out.loc[idx, "split"] = "test"
        with pytest.raises(RuntimeError, match="LEAKAGE DETECTED"):
            verify_no_leakage(out)

    def test_split_is_deterministic(self):
        a = make_splits(_frame(), seed=42)["split"].tolist()
        b = make_splits(_frame(), seed=42)["split"].tolist()
        assert a == b

    def test_different_seed_changes_assignment(self):
        a = make_splits(_frame(), seed=1)["split"].tolist()
        b = make_splits(_frame(), seed=999)["split"].tolist()
        assert a != b

    def test_both_classes_present_in_every_split(self):
        out = make_splits(_frame(n_groups=80), seed=42)
        for split in ("train", "validation", "test"):
            labels = set(out[out["split"] == split]["binary_label"])
            assert labels == {"OK", "Defective"}, f"{split} missing a class: {labels}"

    def test_approximate_ratio_respected(self):
        out = make_splits(_frame(n_groups=200), seed=42)
        frac = out["split"].value_counts(normalize=True)
        assert frac["train"] == pytest.approx(0.70, abs=0.08)
        assert frac["test"] == pytest.approx(0.15, abs=0.08)


class TestDuplicates:
    def test_exact_duplicates_removed(self):
        df = _frame(n_groups=5)
        dup = df.iloc[[0]].copy()
        df2 = pd.concat([df, dup], ignore_index=True)
        kept, report = drop_exact_duplicates(df2)
        assert report["n_exact_duplicates_removed"] == 1
        assert len(kept) == len(df)

    def test_cross_class_duplicates_dropped_entirely(self):
        df = _frame(n_groups=5)
        bad = df.iloc[[0]].copy()
        bad["binary_label"] = "Defective" if df.iloc[0]["binary_label"] == "OK" else "OK"
        df2 = pd.concat([df, bad], ignore_index=True)
        kept, report = drop_exact_duplicates(df2)
        assert report["n_cross_class_groups_dropped"] == 1
        assert df.iloc[0]["content_hash"] not in set(kept["content_hash"])


class TestStatistics:
    def test_statistics_shape(self):
        out = make_splits(_frame(), seed=42)
        stats = split_statistics(out)
        assert stats["total_images"] == len(out)
        for split in ("train", "validation", "test"):
            assert stats[split]["n_images"] > 0
            assert "binary_distribution" in stats[split]


class TestContentGrouping:
    """Content-based grouping — required because this dataset's filenames
    carry no source-casting identity."""

    @staticmethod
    def _make_images(tmp_path, n_sources=6, variants=3):
        """Build images where each source has near-identical variants."""
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(0)
        rows = []
        for s in range(n_sources):
            base = rng.integers(0, 255, (64, 64), dtype=np.uint8)
            for v in range(variants):
                arr = base.copy()
                # a small localised "defect", as the pipeline would paint on
                arr[5:9, 5:9] = (arr[5:9, 5:9].astype(int) + 40 * v) % 255
                cls = "ok" if v == 0 else f"defect{v}"
                d = tmp_path / cls
                d.mkdir(exist_ok=True)
                rel = f"{cls}/src{s}_{v}.png"
                Image.fromarray(arr, "L").save(tmp_path / rel)
                rows.append({
                    "relpath": rel, "filename": f"src{s}_{v}.png",
                    "class_label": cls, "defect_type": cls,
                    "binary_label": "OK" if cls == "ok" else "Defective",
                    "is_defective": cls != "ok", "group_key": f"{s}",
                    "content_hash": f"h{s}{v}", "is_valid": True,
                })
        return pd.DataFrame(rows)

    def test_variants_of_one_source_share_a_group(self, tmp_path):
        from src.dataset import attach_content_groups

        df = self._make_images(tmp_path)
        out = attach_content_groups(df, tmp_path, threshold=0.9)
        for src in range(6):
            keys = set(out[out["relpath"].str.contains(f"src{src}_")]["group_key"])
            assert len(keys) == 1, f"source {src} split across groups {keys}"

    def test_distinct_sources_are_separate_groups(self, tmp_path):
        from src.dataset import attach_content_groups

        df = self._make_images(tmp_path)
        out = attach_content_groups(df, tmp_path, threshold=0.9)
        assert out["group_key"].nunique() == 6

    def test_original_filename_key_is_preserved(self, tmp_path):
        from src.dataset import attach_content_groups

        out = attach_content_groups(self._make_images(tmp_path), tmp_path, threshold=0.9)
        assert "filename_group_key" in out.columns

    def test_split_on_content_groups_is_leakage_free(self, tmp_path):
        from src.dataset import attach_content_groups

        df = attach_content_groups(self._make_images(tmp_path, n_sources=30),
                                   tmp_path, threshold=0.9)
        out = make_splits(df, seed=42)
        assert verify_no_leakage(out)["leakage_free"] is True
        # no source's variants may straddle a split
        for src in range(30):
            splits = set(out[out["relpath"].str.contains(f"src{src}_")]["split"])
            assert len(splits) == 1

    def test_features_are_l2_normalised(self, tmp_path):
        import numpy as np

        from src.dataset import compute_visual_features

        self._make_images(tmp_path, n_sources=3)
        paths = sorted(tmp_path.rglob("*.png"))
        X = compute_visual_features(paths, size=32)
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)

    def test_brightness_shift_does_not_break_matching(self, tmp_path):
        """Contrast normalisation must survive a global exposure change."""
        import numpy as np
        from PIL import Image

        from src.dataset import compute_visual_features

        rng = np.random.default_rng(1)
        base = rng.integers(20, 200, (64, 64), dtype=np.uint8)
        Image.fromarray(base, "L").save(tmp_path / "a.png")
        Image.fromarray(np.clip(base.astype(int) + 40, 0, 255).astype("uint8"),
                        "L").save(tmp_path / "b.png")
        X = compute_visual_features([tmp_path / "a.png", tmp_path / "b.png"], size=32)
        assert float(X[0] @ X[1]) > 0.98

    def test_refuses_oversized_dataset(self, tmp_path):
        from src.dataset import compute_content_groups

        df = self._make_images(tmp_path, n_sources=2, variants=2)
        with pytest.raises(RuntimeError, match="max_images"):
            compute_content_groups(df, tmp_path, max_images=3)
