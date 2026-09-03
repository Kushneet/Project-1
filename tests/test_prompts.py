"""Tests for the frozen prompts and the response parser (Phase 20)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts import (  # noqa: E402
    EVAL_PROMPTS,
    TRAIN_INSTRUCTIONS,
    build_messages,
    build_training_answer,
    get_prompt,
    parse_confidence,
    parse_prediction,
)


class TestPromptRegistry:
    def test_three_prompts_registered(self):
        assert set(EVAL_PROMPTS) == {"prompt_1", "prompt_2", "prompt_3"}

    def test_prompts_are_nonempty(self):
        for pid, text in EVAL_PROMPTS.items():
            assert text.strip(), f"{pid} is empty"

    def test_unknown_prompt_id_raises(self):
        with pytest.raises(KeyError):
            get_prompt("prompt_99")

    def test_prompts_do_not_leak_dataset_labels(self):
        # Injecting the class list would unfairly inflate the baseline.
        leaked = {"cold_shut", "shrinkage", "pinhole", "surface_roughness"}
        for pid, text in EVAL_PROMPTS.items():
            low = text.lower()
            assert not (leaked & {w for w in leaked if w in low}), f"{pid} leaks labels"

    def test_build_messages_shape(self):
        msgs = build_messages("prompt_1")
        assert msgs[0]["role"] == "user"
        kinds = [c["type"] for c in msgs[0]["content"]]
        assert kinds == ["image", "text"]
        assert msgs[0]["content"][1]["text"] == get_prompt("prompt_1")

    def test_training_instructions_present(self):
        assert len(TRAIN_INSTRUCTIONS) >= 5
        assert all(i.strip() for i in TRAIN_INSTRUCTIONS)


class TestBuildTrainingAnswer:
    def test_ok_case(self):
        out = build_training_answer("ok", is_defective=False)
        assert "Classification: OK" in out
        assert "Defect type: None" in out

    def test_defective_with_type(self):
        out = build_training_answer("crack", is_defective=True, defect_type="crack")
        assert "Classification: Defective" in out
        assert "Defect type: crack" in out

    def test_defective_without_type_is_unknown_not_invented(self):
        out = build_training_answer("def", is_defective=True, defect_type=None)
        assert "Defect type: Unknown" in out


class TestParsePrediction:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Classification: OK\nReason: clean surface\nConfidence: 95%", "OK"),
            ("Classification: Defective\nReason: crack\nConfidence: 88%", "Defective"),
            ("Classification: Not OK", "Defective"),
            ("Defect present: Yes\nDefect type: crack", "Defective"),
            ("Defect present: No\nDefect type: Unknown", "OK"),
            ("Prediction: Defective\nEvidence: pitting\nConfidence: 70%", "Defective"),
            ("**Classification:** OK", "OK"),
            ("The casting shows no defect and appears acceptable.", "OK"),
            ("This casting is defective.", "Defective"),
            ("I cannot determine anything from this input.", "Unparseable"),
            ("", "Unparseable"),
        ],
    )
    def test_classification(self, raw, expected):
        assert parse_prediction(raw)["prediction"] == expected

    def test_defect_type_extracted(self):
        r = parse_prediction("Defect present: Yes\nDefect type: porosity\nConfidence: 60%")
        assert r["defect_type"] == "porosity"

    def test_defect_type_unknown_when_absent(self):
        r = parse_prediction("Classification: Defective\nReason: something odd")
        assert r["defect_type"] == "Unknown"

    def test_ok_prediction_gets_none_defect_type(self):
        r = parse_prediction("Classification: OK\nReason: clean")
        assert r["defect_type"] == "None"

    def test_evidence_extracted(self):
        r = parse_prediction("Classification: Defective\nReason: visible crack near rim")
        assert "crack" in r["evidence"]

    def test_format_ok_flag(self):
        assert parse_prediction("Classification: OK").format_ok if False else True
        assert parse_prediction("Classification: OK")["format_ok"] is True
        assert parse_prediction("looks fine to me")["format_ok"] is False

    def test_empty_response_is_never_silently_correct(self):
        r = parse_prediction("   ")
        assert r["prediction"] == "Unparseable"
        assert r["confidence"] is None


class TestParseConfidence:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Confidence: 87%", 87.0),
            ("Confidence: 87", 87.0),
            ("Confidence: 0.87", 87.0),
            ("Confidence: 100%", 100.0),
            ("Confidence: 92 out of 100", 92.0),
            ("Classification: OK\nConfidence: 55%", 55.0),
        ],
    )
    def test_values(self, raw, expected):
        assert parse_confidence(raw) == pytest.approx(expected)

    def test_non_numeric_confidence_returns_none(self):
        assert parse_confidence("Confidence: High") is None

    def test_absent_confidence_returns_none(self):
        assert parse_confidence("Classification: OK") is None

    def test_clamped_to_100(self):
        assert parse_confidence("Confidence: 150%") == 100.0
