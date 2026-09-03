"""Tests for the inference API and config plumbing.

The real VLM is never loaded here — generation is stubbed so the parsing,
fallback and error-handling logic can be tested on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.inference as inference  # noqa: E402
from src.utils import detect_device, load_config, resolve_path, set_seed  # noqa: E402


class _FakeLoaded:
    model_name = "fake/model"
    adapter_path = None
    model = None
    processor = None
    device = "cpu"


@pytest.fixture
def stub_generate(monkeypatch):
    """Replace generate() with a canned response."""
    def _install(response: str):
        monkeypatch.setattr(inference, "generate", lambda *a, **k: response)
    return _install


class TestPredictImage:
    def test_structured_output_keys(self, stub_generate):
        stub_generate("Classification: Defective\nReason: crack\nConfidence: 91%")
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert set(r) >= {"classification", "defect_type", "confidence",
                          "evidence", "raw_response"}

    def test_defective_parsed(self, stub_generate):
        stub_generate("Classification: Defective\nReason: visible crack\nConfidence: 91%")
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert r["classification"] == "Defective"
        assert r["confidence"] == 91.0
        assert "crack" in r["evidence"]

    def test_ok_parsed(self, stub_generate):
        stub_generate("Classification: OK\nReason: clean surface\nConfidence: 80%")
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert r["classification"] == "OK"
        assert r["defect_type"] == "None"

    def test_confidence_is_flagged_uncalibrated(self, stub_generate):
        stub_generate("Classification: OK\nConfidence: 99%")
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert "not calibrated" in r["confidence_note"]

    def test_garbage_response_is_unparseable_not_guessed(self, stub_generate):
        stub_generate("The weather today is pleasant.")
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert r["classification"] == "Unparseable"

    def test_raw_response_preserved(self, stub_generate):
        raw = "Classification: Defective\nReason: pitting everywhere"
        stub_generate(raw)
        r = inference.predict_image("dummy.jpg", loaded=_FakeLoaded())
        assert r["raw_response"] == raw


class TestConfig:
    def test_config_loads(self):
        cfg = load_config()
        for section in ("project", "data", "model", "training", "lora",
                        "baseline", "evaluation", "split"):
            assert section in cfg, f"missing config section: {section}"

    def test_model_is_the_documented_choice(self):
        assert load_config()["model"]["model_name"] == "Qwen/Qwen3-VL-4B-Instruct"

    def test_split_ratios_sum_to_one(self):
        s = load_config()["split"]
        total = s["train_ratio"] + s["validation_ratio"] + s["test_ratio"]
        assert abs(total - 1.0) < 1e-9

    def test_seed_is_fixed_everywhere(self):
        cfg = load_config()
        assert cfg["project"]["seed"] == cfg["split"]["seed"] == cfg["inference"]["seed"]

    def test_inference_is_deterministic_by_default(self):
        # Sampling would make baseline/fine-tuned runs non-reproducible.
        assert load_config()["inference"]["do_sample"] is False

    def test_learning_rate_parses_as_float(self):
        assert isinstance(float(load_config()["training"]["learning_rate"]), float)

    def test_lora_targets_declared(self):
        assert len(load_config()["lora"]["target_modules"]) > 0


class TestUtils:
    def test_resolve_relative_path(self):
        assert resolve_path("data/raw").is_absolute()

    def test_resolve_absolute_path_unchanged(self, tmp_path):
        assert resolve_path(tmp_path) == tmp_path

    def test_detect_device_returns_known_value(self):
        assert detect_device() in {"cuda", "mps", "cpu"}

    def test_set_seed_is_reproducible(self):
        import random

        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        assert a == [random.random() for _ in range(5)]


class TestCheckpointLoading:
    def test_missing_adapter_falls_back_to_base_with_warning(self, stub_generate, caplog):
        stub_generate("Classification: OK")
        cfg = load_config()
        adapter = resolve_path(cfg["training"]["output_dir"])
        if adapter.exists():
            pytest.skip("a real adapter exists; fallback path not exercised")

        called: dict[str, object] = {}

        def fake_load_model(**kwargs):
            called.update(kwargs)
            return _FakeLoaded()

        import src.inference as inf

        original = inf.load_model
        inf.load_model = fake_load_model
        try:
            inf.predict_image("dummy.jpg", use_finetuned=True)
        finally:
            inf.load_model = original

        assert called["adapter_path"] is None, \
            "must fall back to the base model rather than silently claiming fine-tuned"
