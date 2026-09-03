"""Model loading, generation and the public prediction API (Phase 13).

Written against the current Transformers API for Qwen3-VL:
``Qwen3VLForConditionalGeneration`` + ``AutoProcessor``, with
``processor.apply_chat_template(..., tokenize=True, return_dict=True)``.

Heavy imports (torch, transformers) are deliberately performed inside the
functions so that the dataset-analysis and test paths stay importable on a
machine with no ML stack installed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_messages, get_prompt, parse_prediction
from .utils import detect_device, get_logger, load_config, resolve_path

LOG = get_logger("inference")


@dataclass
class LoadedModel:
    """A model plus its processor and the settings used to load it."""

    model: Any
    processor: Any
    device: str
    model_name: str
    adapter_path: str | None = None


def load_model(
    model_name: str,
    adapter_path: str | Path | None = None,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> LoadedModel:
    """Load the base VLM, optionally applying a trained LoRA adapter.

    ``adapter_path=None`` gives the untouched pretrained model — this is what
    the Phase-3/4 baseline must use.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = detect_device()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)
    if device != "cuda" and torch_dtype is torch.bfloat16:
        # bf16 is unreliable outside CUDA; fp32 on CPU, fp16 on MPS.
        torch_dtype = torch.float16 if device == "mps" else torch.float32
        LOG.info("Device %s: using %s instead of bfloat16", device, torch_dtype)

    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"

    LOG.info("Loading %s on %s (dtype=%s, 4bit=%s)", model_name, device,
             torch_dtype, load_in_4bit and device == "cuda")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        path = str(resolve_path(adapter_path))
        LOG.info("Applying LoRA adapter from %s", path)
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()  # fold LoRA in for faster inference

    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    proc_kwargs: dict[str, Any] = {}
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

    return LoadedModel(
        model=model, processor=processor, device=device,
        model_name=model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
    )


def generate(
    loaded: LoadedModel,
    image_path: str | Path,
    prompt_id: str = "prompt_1",
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> str:
    """Run one image + one prompt through the model and return raw text.

    Decoding is greedy by default so baseline and fine-tuned runs are
    reproducible and directly comparable.
    """
    import torch
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as im:
        image = im.convert("RGB")  # the vision tower expects 3 channels

    messages = build_messages(prompt_id, image=image)
    inputs = loaded.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)  # not accepted by generate()
    inputs = {k: (v.to(loaded.model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    # Strip the prompt tokens so only the completion is decoded.
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    return loaded.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def predict_image(
    image_path: str | Path,
    loaded: LoadedModel | None = None,
    prompt_id: str = "prompt_1",
    config_path: str | Path | None = None,
    use_finetuned: bool = True,
) -> dict[str, Any]:
    """Predict OK/Defective for a single image.

    Returns ``classification``, ``defect_type``, ``confidence``, ``evidence``
    and ``raw_response``. ``confidence`` is the model's own self-reported
    number, NOT a calibrated probability.
    """
    cfg = load_config(config_path)
    if loaded is None:
        adapter = cfg["training"]["output_dir"] if use_finetuned else None
        if adapter is not None and not resolve_path(adapter).exists():
            LOG.warning("No fine-tuned adapter at %s — falling back to the base model. "
                        "This prediction is NOT from the fine-tuned model.", adapter)
            adapter = None
        loaded = load_model(
            model_name=cfg["model"]["model_name"],
            adapter_path=adapter,
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            min_pixels=cfg["model"].get("min_pixels"),
            max_pixels=cfg["model"].get("max_pixels"),
        )

    raw = generate(
        loaded, image_path, prompt_id=prompt_id,
        max_new_tokens=cfg["model"]["max_new_tokens"],
        do_sample=cfg["inference"]["do_sample"],
    )
    parsed = parse_prediction(raw)
    return {
        "image_path": str(image_path),
        "classification": parsed["prediction"],
        "defect_type": parsed["defect_type"],
        "confidence": parsed["confidence"],
        "confidence_note": "model-reported, not calibrated",
        "evidence": parsed["evidence"],
        "raw_response": raw,
        "prompt_id": prompt_id,
        "model": loaded.model_name,
        "adapter": loaded.adapter_path,
    }


def main() -> None:
    """CLI: python -m src.inference --image path/to/image.jpg"""
    parser = argparse.ArgumentParser(description="Predict OK/Defective for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-id", default="prompt_1")
    parser.add_argument("--base-model", action="store_true",
                        help="Use the untuned base model instead of the fine-tuned one")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    result = predict_image(
        args.image, prompt_id=args.prompt_id,
        config_path=args.config, use_finetuned=not args.base_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
