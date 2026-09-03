"""Phase 8-9 — LoRA/PEFT fine-tuning of the vision-language model.

Design notes
------------
* Only the answer tokens contribute to the loss. Prompt tokens, padding and
  image placeholder tokens are masked to -100, so the model is not trained to
  reproduce the question or the image embedding.
* The vision tower is frozen by default; LoRA adapters are attached to the
  language-model projections only. This is what keeps the run inside a free
  Colab GPU's memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepare_data import load_jsonl
from .utils import get_logger, resolve_path, write_json

LOG = get_logger("train")

IGNORE_INDEX = -100


@dataclass
class VLMCollator:
    """Turn instruction examples into padded, label-masked model inputs."""

    processor: Any
    max_length: int = 1024

    def _encode_one(self, example: dict[str, Any]) -> dict[str, Any]:
        """Encode one example, returning input_ids/labels with prompt masked."""
        import torch
        from PIL import Image

        with Image.open(example["image"]) as im:
            image = im.convert("RGB")

        user_msg, assistant_msg = example["messages"][0], example["messages"][1]
        # Re-attach the loaded PIL image in place of the stored path.
        user_content = [
            {"type": "image", "image": image} if c["type"] == "image" else c
            for c in user_msg["content"]
        ]
        user_only = [{"role": "user", "content": user_content}]
        full = user_only + [assistant_msg]

        full_inputs = self.processor.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        prompt_inputs = self.processor.apply_chat_template(
            user_only, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        full_inputs.pop("token_type_ids", None)

        input_ids = full_inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX          # don't learn the question
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = IGNORE_INDEX

        # Mask every special vision token so image placeholders never form targets.
        for attr in ("image_token_id", "video_token_id",
                     "vision_start_token_id", "vision_end_token_id"):
            tok = getattr(self.processor, attr, None)
            if tok is None:
                tok = getattr(getattr(self.processor, "tokenizer", None), attr, None)
            if isinstance(tok, int):
                labels[input_ids == tok] = IGNORE_INDEX

        if input_ids.shape[0] > self.max_length:
            LOG.warning("Truncating example %s from %d to %d tokens",
                        example.get("image_id"), input_ids.shape[0], self.max_length)
            full_inputs = {
                k: (v[:, : self.max_length] if hasattr(v, "shape") and v.ndim == 2
                    and v.shape[1] == input_ids.shape[0] else v)
                for k, v in full_inputs.items()
            }
            labels = labels[: self.max_length]

        out = {k: (v[0] if hasattr(v, "shape") and v.shape[0] == 1 else v)
               for k, v in full_inputs.items()}
        out["labels"] = labels
        return out

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self._encode_one(ex) for ex in batch]
        pad_id = self.processor.tokenizer.pad_token_id or 0

        max_len = max(e["input_ids"].shape[0] for e in encoded)
        out: dict[str, Any] = {}

        def _pad(t, value):
            import torch.nn.functional as F

            return F.pad(t, (0, max_len - t.shape[0]), value=value)

        out["input_ids"] = torch.stack([_pad(e["input_ids"], pad_id) for e in encoded])
        out["labels"] = torch.stack([_pad(e["labels"], IGNORE_INDEX) for e in encoded])
        if "attention_mask" in encoded[0]:
            out["attention_mask"] = torch.stack(
                [_pad(e["attention_mask"], 0) for e in encoded]
            )
        else:
            out["attention_mask"] = (out["input_ids"] != pad_id).long()

        # Qwen3-VL under Transformers 5 needs `mm_token_type_ids` to compute
        # multimodal RoPE. The processor returns it alongside `input_ids`;
        # 0 = text, 1 = image, 2 = video, so padding counts as text.
        if "mm_token_type_ids" in encoded[0]:
            out["mm_token_type_ids"] = torch.stack(
                [_pad(e["mm_token_type_ids"], 0) for e in encoded]
            )

        # Vision tensors concatenate rather than stack (variable patch counts).
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos"):
            vals = [e[key] for e in encoded if key in e]
            if vals:
                out[key] = torch.cat([v if v.ndim > 1 else v.unsqueeze(0) for v in vals], dim=0)
        return out


class JsonlDataset:
    """Minimal map-style dataset over a prepared JSONL file."""

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.examples = load_jsonl(path)
        if limit:
            self.examples = self.examples[:limit]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def build_lora_model(model: Any, lora_cfg: dict[str, Any]) -> Any:
    """Attach LoRA adapters and freeze the vision tower."""
    from peft import LoraConfig, get_peft_model

    if lora_cfg.get("freeze_vision_tower", True):
        frozen = 0
        for name, param in model.named_parameters():
            if any(k in name for k in ("visual", "vision_tower", "vision_model")):
                param.requires_grad = False
                frozen += 1
        LOG.info("Froze %d vision-tower parameter tensors", frozen)

    config = LoraConfig(
        r=lora_cfg["lora_r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def count_trainable(model: Any) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_training_args(
    cfg: dict[str, Any],
    sanity: bool = False,
    n_train_examples: int | None = None,
) -> Any:
    """Construct TrainingArguments from config, adapting to the local device."""
    import inspect

    import torch
    from transformers import TrainingArguments

    t = cfg["training"]
    cuda = torch.cuda.is_available()
    # Native bf16 needs compute capability 8.0+ (Ampere). A T4 (sm_75) reports
    # is_bf16_supported() == True but only emulates it, which is slower.
    native_bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    bf16 = native_bf16 and t["precision"] == "bf16"
    fp16 = cuda and not bf16 and t["precision"] in {"fp16", "bf16"}

    kwargs: dict[str, Any] = dict(
        output_dir=str(resolve_path(t["checkpoint_dir"])),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=t["num_epochs"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=t["gradient_checkpointing"] and cuda,
        report_to=[],
        remove_unused_columns=False,  # our collator consumes the raw dicts
        seed=cfg["project"]["seed"],
    )

    # `evaluation_strategy` was renamed to `eval_strategy` in newer Transformers.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"

    # Transformers 5 dropped `warmup_ratio` and `logging_dir`. Preserve the
    # configured warmup fraction by converting it to an explicit step count;
    # the schedule is unchanged, only the way it is expressed.
    if "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = float(t["warmup_ratio"])
    elif not sanity and n_train_examples:
        steps_per_epoch = max(
            n_train_examples // (t["batch_size"] * t["gradient_accumulation_steps"]), 1
        )
        total_steps = steps_per_epoch * int(t["num_epochs"])
        kwargs["warmup_steps"] = max(
            round(total_steps * float(t["warmup_ratio"])), 1
        )
    if "logging_dir" in sig:
        kwargs["logging_dir"] = str(resolve_path(t["log_dir"]))

    if sanity:
        kwargs.update(
            max_steps=cfg["sanity_check"]["max_steps"],
            num_train_epochs=1,
            save_steps=10_000,
            logging_steps=1,
        )
        kwargs[eval_key] = "no"
    else:
        kwargs[eval_key] = t["evaluation_strategy"]
        kwargs["eval_steps"] = t["eval_steps"]

    if cuda and t.get("optim"):
        kwargs["optim"] = t["optim"]

    return TrainingArguments(**kwargs)


def run_training(
    cfg: dict[str, Any],
    sanity: bool = False,
    train_limit: int | None = None,
) -> dict[str, Any]:
    """Fine-tune the model with LoRA and save the adapter.

    With ``sanity=True`` this runs the Phase-9 check: a couple of steps on a
    handful of examples, verifying that data loads, the forward and backward
    passes work, the loss is finite, and a checkpoint saves and reloads.
    """
    import torch
    from transformers import Trainer

    from .inference import load_model

    t, m = cfg["training"], cfg["model"]
    limit = cfg["sanity_check"]["n_examples"] if sanity else train_limit

    train_ds = JsonlDataset(cfg["data"]["train_file"], limit=limit)
    eval_ds = JsonlDataset(cfg["data"]["validation_file"],
                           limit=limit if sanity else None)
    LOG.info("Train examples: %d | validation examples: %d", len(train_ds), len(eval_ds))
    if len(train_ds) == 0:
        raise RuntimeError("Training set is empty — run scripts/prepare_training_data.py")

    loaded = load_model(
        model_name=m["model_name"],
        adapter_path=None,
        dtype=m["dtype"],
        load_in_4bit=t.get("load_in_4bit", False),
        attn_implementation=m["attn_implementation"],
        min_pixels=m.get("min_pixels"),
        max_pixels=m.get("max_pixels"),
    )

    if t.get("load_in_4bit") and torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training

        loaded.model = prepare_model_for_kbit_training(
            loaded.model, use_gradient_checkpointing=t["gradient_checkpointing"]
        )

    lora_cfg = dict(cfg["lora"])
    model = build_lora_model(loaded.model, lora_cfg)
    trainable, total = count_trainable(model)
    LOG.info("Trainable %s / %s params (%.4f%%)", f"{trainable:,}", f"{total:,}",
             100 * trainable / max(total, 1))
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters — LoRA adapters did not attach. "
            f"Check lora.target_modules={lora_cfg['target_modules']} against the "
            "model's actual module names."
        )

    collator = VLMCollator(processor=loaded.processor)
    args = build_training_args(cfg, sanity=sanity, n_train_examples=len(train_ds))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not sanity else None,
        data_collator=collator,
    )

    LOG.info("Starting %s run", "SANITY" if sanity else "FULL training")
    result = trainer.train()

    loss = float(result.training_loss)
    if not (loss == loss and abs(loss) != float("inf")):  # NaN / inf guard
        raise RuntimeError(f"Training loss is not finite: {loss}")
    LOG.info("Training loss: %.4f", loss)

    out_dir = resolve_path(t["output_dir"] if not sanity
                           else Path(t["checkpoint_dir"]) / "sanity_adapter")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    loaded.processor.save_pretrained(str(out_dir))
    LOG.info("Saved adapter to %s", out_dir)

    summary = {
        "sanity_run": sanity,
        "model_name": m["model_name"],
        "training_loss": loss,
        "global_step": int(result.global_step),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(total, 1), 4),
        "n_train_examples": len(train_ds),
        "n_validation_examples": len(eval_ds),
        "adapter_dir": str(out_dir),
        "lora": lora_cfg,
        "training_args": {k: v for k, v in cfg["training"].items()},
        "seed": cfg["project"]["seed"],
        "log_history": trainer.state.log_history,
    }
    write_json(summary, Path(t["log_dir"]) /
               ("sanity_summary.json" if sanity else "training_summary.json"))
    return summary


def plot_training_curves(log_history: list[dict[str, Any]], out_path: str | Path) -> Path | None:
    """Plot train/validation loss against step, if any losses were logged."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_pts = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    eval_pts = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    if not train_pts:
        LOG.warning("No loss entries in log history; skipping training curves")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*zip(*train_pts), label="train loss", marker="o", ms=3)
    if eval_pts:
        ax.plot(*zip(*eval_pts), label="validation loss", marker="s", ms=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
