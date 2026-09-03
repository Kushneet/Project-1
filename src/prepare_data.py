"""Phase 6 — convert the image index into multimodal instruction examples.

Every answer is derived from the dataset's ground-truth label. No label is
invented, and no defect type is asserted that the folder structure does not
support.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .prompts import TRAIN_INSTRUCTIONS, build_training_answer
from .utils import get_logger, resolve_path, write_json

LOG = get_logger("prepare_data")


def build_example(
    row: pd.Series,
    image_root: Path,
    instruction: str,
    include_defect_type: bool,
) -> dict[str, Any]:
    """Build one instruction example in a chat-message format.

    ``include_defect_type`` is driven by whether the dataset actually carries
    reliable defect-type labels — never assumed.
    """
    defect_type = row["defect_type"] if include_defect_type else None
    answer = build_training_answer(
        label=str(row["class_label"]),
        is_defective=bool(row["is_defective"]),
        defect_type=str(defect_type) if defect_type is not None else None,
    )
    image_path = str((image_root / row["relpath"]).resolve())
    return {
        "image_id": Path(row["relpath"]).stem,
        "image": image_path,
        "relpath": row["relpath"],
        "ground_truth": row["binary_label"],
        "defect_type": row["defect_type"],
        "group_key": row["group_key"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path},
                            {"type": "text", "text": instruction}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


def build_split_examples(
    df: pd.DataFrame,
    image_root: Path,
    include_defect_type: bool,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build examples for one split, cycling instruction templates."""
    rng = random.Random(seed)
    examples = []
    for _, row in df.iterrows():
        instruction = rng.choice(TRAIN_INSTRUCTIONS)
        examples.append(build_example(row, image_root, instruction, include_defect_type))
    return examples


def write_jsonl(examples: list[dict[str, Any]], path: str | Path) -> Path:
    """Write examples one JSON object per line."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    LOG.info("Wrote %d examples to %s", len(examples), out)
    return out


def load_jsonl(
    path: str | Path,
    image_root: str | Path | None = None,
    repair_image_paths: bool = True,
) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts.

    The ``image`` field was written as an absolute path on whichever machine
    generated the file, so it does not survive being moved to another machine
    (e.g. a Colab runtime). Each example also carries ``relpath``, which is
    machine-independent, so by default the absolute path is rebuilt here from
    the dataset root actually in use. Pass ``repair_image_paths=False`` to keep
    whatever was stored.
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/prepare_training_data.py")
    with p.open("r", encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh if line.strip()]

    if not repair_image_paths:
        return examples

    if image_root is None:
        from .data_analysis import find_dataset_root
        from .utils import load_config

        image_root = find_dataset_root(load_config()["data"]["raw_dir"])
    root = resolve_path(image_root)

    repaired = 0
    for ex in examples:
        rel = ex.get("relpath")
        if not rel:
            continue
        rebuilt = str(root / rel)
        if ex.get("image") != rebuilt:
            ex["image"] = rebuilt
            repaired += 1
        msgs = ex.get("messages") or []
        for msg in msgs:
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    part["image"] = rebuilt
    if repaired:
        LOG.info("Rebuilt %d/%d image paths against %s", repaired, len(examples), root)
    return examples


def dataset_statistics(
    splits: dict[str, list[dict[str, Any]]],
    include_defect_type: bool,
) -> dict[str, Any]:
    """Summarise the prepared instruction dataset."""
    stats: dict[str, Any] = {
        "defect_type_supervision_enabled": include_defect_type,
        "instruction_templates": TRAIN_INSTRUCTIONS,
        "n_instruction_templates": len(TRAIN_INSTRUCTIONS),
    }
    for name, examples in splits.items():
        gt = pd.Series([e["ground_truth"] for e in examples])
        dt = pd.Series([e["defect_type"] for e in examples])
        stats[name] = {
            "n_examples": len(examples),
            "binary_distribution": gt.value_counts().to_dict(),
            "defect_type_distribution": dt.value_counts().to_dict(),
            "n_unique_groups": len({e["group_key"] for e in examples}),
        }
    return stats


def prepare(
    splits_csv: str | Path,
    image_root: str | Path,
    out_dir: str | Path,
    stats_path: str | Path,
    include_defect_type: bool,
    seed: int = 42,
) -> dict[str, Any]:
    """Build train/validation/test JSONL files and a statistics JSON."""
    # group_key must stay a string: pandas would otherwise coerce "0003" -> 3.
    df = pd.read_csv(resolve_path(splits_csv), dtype={"group_key": str})
    root = resolve_path(image_root)

    built: dict[str, list[dict[str, Any]]] = {}
    for name, filename in (
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("test", "test.jsonl"),
    ):
        sub = df[df["split"] == name]
        examples = build_split_examples(sub, root, include_defect_type, seed=seed)
        write_jsonl(examples, Path(out_dir) / filename)
        built[name] = examples

    # Hard guarantee: no image may appear in more than one split.
    seen: dict[str, str] = {}
    for name, examples in built.items():
        for ex in examples:
            if ex["relpath"] in seen:
                raise RuntimeError(
                    f"LEAKAGE: {ex['relpath']} appears in both {seen[ex['relpath']]} "
                    f"and {name}"
                )
            seen[ex["relpath"]] = name

    stats = dataset_statistics(built, include_defect_type)
    write_json(stats, stats_path)
    return stats
