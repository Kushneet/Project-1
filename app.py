#!/usr/bin/env python3
"""Phase 14 — Gradio demo for casting defect detection.

    python app.py
    python app.py --share          # public link (useful from Colab)
    python app.py --base-model     # demo the un-finetuned model instead

The model is loaded once, lazily, on the first prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from src.prompts import EVAL_PROMPTS
from src.utils import get_logger, load_config, resolve_path

LOG = get_logger("app")

TITLE = "Casting Defect Detection using a Fine-Tuned Vision-Language Model"

DISCLAIMER = (
    "This is a research prototype for visual inspection and should not be "
    "treated as a certified industrial quality-control system."
)

_STATE: dict[str, Any] = {"model": None, "cfg": None, "use_finetuned": True}


def _get_model():
    """Load the model once and cache it for subsequent requests."""
    if _STATE["model"] is None:
        from src.inference import load_model

        cfg = _STATE["cfg"]
        adapter = cfg["training"]["output_dir"] if _STATE["use_finetuned"] else None
        if adapter and not resolve_path(adapter).exists():
            LOG.warning("No adapter at %s — demoing the BASE model.", adapter)
            adapter = None
            _STATE["use_finetuned"] = False
        LOG.info("Loading model (fine-tuned=%s)...", _STATE["use_finetuned"])
        _STATE["model"] = load_model(
            model_name=cfg["model"]["model_name"],
            adapter_path=adapter,
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            min_pixels=cfg["model"].get("min_pixels"),
            max_pixels=cfg["model"].get("max_pixels"),
        )
    return _STATE["model"]


def predict(image, prompt_id: str) -> tuple[str, str, str, str, str]:
    """Run one prediction and format it for the UI."""
    if image is None:
        return "—", "—", "—", "Please upload a casting image.", ""

    import tempfile

    from src.inference import predict_image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        path = Path(tmp.name)

    try:
        result = predict_image(path, loaded=_get_model(), prompt_id=prompt_id)
    except Exception as exc:  # noqa: BLE001 - surface errors in the UI, don't crash
        LOG.exception("Prediction failed")
        return "Error", "—", "—", f"Prediction failed: {exc}", ""
    finally:
        path.unlink(missing_ok=True)

    classification = result["classification"]
    if classification == "Unparseable":
        classification = "Unparseable (model did not follow the output format)"

    defect_type = result["defect_type"] or "Unknown"
    conf = result["confidence"]
    confidence = "Not reported" if conf is None else f"{conf:.0f}%  (model-reported, not calibrated)"
    evidence = result["evidence"] or "(the model gave no explicit evidence)"
    return classification, defect_type, confidence, evidence, result["raw_response"]


def build_interface() -> gr.Blocks:
    """Assemble the Gradio UI."""
    cfg = _STATE["cfg"]
    mode = "fine-tuned" if _STATE["use_finetuned"] else "BASE (not fine-tuned)"

    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(
            f"**Model:** `{cfg['model']['model_name']}` &nbsp;|&nbsp; **Mode:** {mode}\n\n"
            "Upload a casting image to check whether it is OK or defective."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Upload casting image",
                                    height=340, sources=["upload", "clipboard"])
                prompt_in = gr.Dropdown(
                    choices=sorted(EVAL_PROMPTS),
                    value=cfg["baseline"]["primary_prompt_id"],
                    label="Prompt",
                    info="The same prompts used in the baseline and evaluation.",
                )
                run = gr.Button("Inspect casting", variant="primary")
            with gr.Column(scale=1):
                out_class = gr.Textbox(label="Classification (OK / Defective)")
                out_type = gr.Textbox(label="Defect Type")
                out_conf = gr.Textbox(label="Model-Reported Confidence")
                out_ev = gr.Textbox(label="Visual Evidence", lines=3)
                out_raw = gr.Textbox(label="Raw Model Response", lines=6)

        run.click(predict, inputs=[image_in, prompt_in],
                  outputs=[out_class, out_type, out_conf, out_ev, out_raw])

        gr.Markdown(
            f"---\n**Disclaimer:** {DISCLAIMER}\n\n"
            "Confidence is the model's own self-reported number, not a calibrated "
            "probability. See `reports/final_report.md` for measured performance "
            "and known limitations."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--base-model", action="store_true",
                        help="Demo the base model instead of the fine-tuned one")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _STATE["cfg"] = load_config(args.config)
    _STATE["use_finetuned"] = not args.base_model
    build_interface().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
