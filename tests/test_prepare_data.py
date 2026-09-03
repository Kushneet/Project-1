"""Tests for instruction-example construction and the frozen eval subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import select_eval_subset  # noqa: E402
from src.prepare_data import (  # noqa: E402
    build_example,
    build_split_examples,
    dataset_statistics,
    load_jsonl,
    write_jsonl,
)


def _row(cls="crack", defective=True, rel="crack/a_0001.jpeg"):
    return pd.Series({
        "relpath": rel, "filename": Path(rel).name, "class_label": cls,
        "defect_type": cls, "binary_label": "Defective" if defective else "OK",
        "is_defective": defective, "group_key": "0001",
    })


class TestBuildExample:
    def test_structure(self, tmp_path):
        ex = build_example(_row(), tmp_path, "Classify this casting image.", True)
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][1]["role"] == "assistant"
        assert ex["ground_truth"] == "Defective"

    def test_answer_uses_ground_truth_label(self, tmp_path):
        ex = build_example(_row("porosity"), tmp_path, "q", True)
        answer = ex["messages"][1]["content"][0]["text"]
        assert "Classification: Defective" in answer
        assert "Defect type: porosity" in answer

    def test_defect_type_suppressed_when_unsupported(self, tmp_path):
        ex = build_example(_row("porosity"), tmp_path, "q", include_defect_type=False)
        answer = ex["messages"][1]["content"][0]["text"]
        # Must fall back to Unknown, never invent a type the labels don't support.
        assert "Defect type: Unknown" in answer
        assert "porosity" not in answer

    def test_ok_row_never_labelled_defective(self, tmp_path):
        ex = build_example(_row("ok", defective=False, rel="ok/b_2.jpeg"), tmp_path, "q", True)
        answer = ex["messages"][1]["content"][0]["text"]
        assert "Classification: OK" in answer
        assert "Defective" not in answer

    def test_image_path_is_absolute(self, tmp_path):
        ex = build_example(_row(), tmp_path, "q", True)
        assert Path(ex["image"]).is_absolute()


class TestSplitExamples:
    def test_deterministic_for_same_seed(self, tmp_path):
        df = pd.DataFrame([_row(rel=f"crack/a_{i:04d}.jpeg") for i in range(20)])
        a = build_split_examples(df, tmp_path, True, seed=42)
        b = build_split_examples(df, tmp_path, True, seed=42)
        assert [e["messages"][0]["content"][1]["text"] for e in a] == \
               [e["messages"][0]["content"][1]["text"] for e in b]

    def test_multiple_instruction_templates_used(self, tmp_path):
        df = pd.DataFrame([_row(rel=f"crack/a_{i:04d}.jpeg") for i in range(60)])
        exs = build_split_examples(df, tmp_path, True, seed=42)
        assert len({e["messages"][0]["content"][1]["text"] for e in exs}) >= 4


class TestJsonlRoundTrip:
    def test_write_then_read(self, tmp_path):
        exs = [{"image_id": "a", "ground_truth": "OK", "defect_type": "ok",
                "group_key": "1", "messages": []}]
        p = write_jsonl(exs, tmp_path / "x.jsonl")
        assert load_jsonl(p) == exs

    def test_missing_file_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="prepare_training_data"):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_statistics(self):
        splits = {"train": [{"ground_truth": "OK", "defect_type": "ok", "group_key": "1"},
                            {"ground_truth": "Defective", "defect_type": "crack",
                             "group_key": "2"}]}
        s = dataset_statistics(splits, True)
        assert s["train"]["n_examples"] == 2
        assert s["train"]["n_unique_groups"] == 2
        assert s["defect_type_supervision_enabled"] is True


class TestEvalSubsetFreeze:
    @staticmethod
    def _examples(n=100):
        return [{"relpath": f"c/img_{i:03d}.jpg", "image": f"/x/img_{i:03d}.jpg",
                 "image_id": f"img_{i:03d}",
                 "ground_truth": "Defective" if i % 2 else "OK",
                 "defect_type": "crack" if i % 2 else "ok"} for i in range(n)]

    def test_subset_written_and_balanced(self, tmp_path):
        p = tmp_path / "subset.json"
        chosen = select_eval_subset(self._examples(), n=20, subset_path=p)
        assert len(chosen) == 20
        assert p.exists()
        labels = pd.Series([e["ground_truth"] for e in chosen]).value_counts()
        assert set(labels.index) == {"OK", "Defective"}
        assert abs(labels["OK"] - labels["Defective"]) <= 1

    def test_subset_is_reused_not_resampled(self, tmp_path):
        p = tmp_path / "subset.json"
        first = select_eval_subset(self._examples(), n=20, subset_path=p)
        second = select_eval_subset(self._examples(), n=50, subset_path=p, reuse=True)
        assert [e["relpath"] for e in first] == [e["relpath"] for e in second], \
            "the frozen subset must not change between baseline and evaluation"

    def test_changed_split_is_detected(self, tmp_path):
        p = tmp_path / "subset.json"
        select_eval_subset(self._examples(), n=20, subset_path=p)
        # Simulate a re-split that dropped the original images.
        with pytest.raises(RuntimeError, match="split changed|not be comparable"):
            select_eval_subset(self._examples(200)[100:], n=20, subset_path=p, reuse=True)

    def test_none_uses_full_set(self, tmp_path):
        chosen = select_eval_subset(self._examples(30), n=None,
                                    subset_path=tmp_path / "s.json")
        assert len(chosen) == 30

    def test_deterministic_for_seed(self, tmp_path):
        a = select_eval_subset(self._examples(), n=16, seed=7,
                               subset_path=tmp_path / "a.json")
        b = select_eval_subset(self._examples(), n=16, seed=7,
                               subset_path=tmp_path / "b.json")
        assert [e["relpath"] for e in a] == [e["relpath"] for e in b]


class TestPortableImagePaths:
    """test.jsonl stores absolute paths; they must be rebuilt on another machine."""

    @staticmethod
    def _write(tmp_path, stale_root="/some/other/machine"):
        import json

        root = tmp_path / "imgs"
        (root / "crack").mkdir(parents=True)
        (root / "crack" / "a.jpg").write_bytes(b"x")
        rec = {
            "image_id": "a",
            "relpath": "crack/a.jpg",
            "image": f"{stale_root}/crack/a.jpg",   # path from a different machine
            "ground_truth": "Defective",
            "defect_type": "crack",
            "group_key": "1",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": f"{stale_root}/crack/a.jpg"},
                    {"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
        }
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps(rec) + "\n")
        return f, root

    def test_stale_absolute_path_is_rebuilt(self, tmp_path):
        from src.prepare_data import load_jsonl

        f, root = self._write(tmp_path)
        ex = load_jsonl(f, image_root=root)[0]
        assert ex["image"] == str(root / "crack/a.jpg")
        assert Path(ex["image"]).exists()

    def test_message_content_path_is_rebuilt_too(self, tmp_path):
        from src.prepare_data import load_jsonl

        f, root = self._write(tmp_path)
        ex = load_jsonl(f, image_root=root)[0]
        img_part = ex["messages"][0]["content"][0]
        assert img_part["image"] == str(root / "crack/a.jpg"), \
            "the chat-message image path must be rebuilt as well"

    def test_repair_can_be_disabled(self, tmp_path):
        from src.prepare_data import load_jsonl

        f, root = self._write(tmp_path)
        ex = load_jsonl(f, image_root=root, repair_image_paths=False)[0]
        assert ex["image"].startswith("/some/other/machine")
