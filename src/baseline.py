"""Phase 4 — base-model baseline inference.

The evaluation subset selected here is written to disk and reused verbatim by
the fine-tuned evaluation (Phase 10). Baseline and fine-tuned runs therefore
see identical images and identical prompts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .prompts import canonicalize_defect_type, get_prompt, parse_prediction
from .utils import get_logger, resolve_path, set_seed, write_json

LOG = get_logger("baseline")

EVAL_SUBSET_FILE = "results/baseline/eval_subset.json"


def select_eval_subset(
    test_examples: list[dict[str, Any]],
    n: int | None,
    seed: int = 42,
    subset_path: str | Path = EVAL_SUBSET_FILE,
    reuse: bool = True,
) -> list[dict[str, Any]]:
    """Choose a fixed, class-balanced evaluation subset — once.

    If a subset file already exists it is reused, so the fine-tuned model is
    never evaluated on a different sample than the baseline.
    """
    path = resolve_path(subset_path)
    by_relpath = {e["relpath"]: e for e in test_examples}

    if reuse and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        missing = [r for r in saved["relpaths"] if r not in by_relpath]
        if missing:
            raise RuntimeError(
                f"{len(missing)} images from the saved eval subset are absent from the "
                f"current test split (e.g. {missing[:3]}). The split changed — "
                "baseline and fine-tuned results would not be comparable."
            )
        LOG.info("Reusing existing eval subset of %d images from %s",
                 len(saved["relpaths"]), path)
        return [by_relpath[r] for r in saved["relpaths"]]

    if n is None or n >= len(test_examples):
        chosen = list(test_examples)
        LOG.info("Using the entire test split: %d images", len(chosen))
    else:
        import random

        rng = random.Random(seed)
        # Sample each class proportionally, then top up, so both classes appear.
        by_class: dict[str, list[dict[str, Any]]] = {}
        for e in test_examples:
            by_class.setdefault(e["ground_truth"], []).append(e)
        per_class = max(1, n // max(len(by_class), 1))
        chosen = []
        for label, items in sorted(by_class.items()):
            pool = sorted(items, key=lambda x: x["relpath"])
            rng.shuffle(pool)
            chosen.extend(pool[:min(per_class, len(pool))])
        # Fill any shortfall from whatever remains.
        if len(chosen) < n:
            taken = {e["relpath"] for e in chosen}
            rest = [e for e in test_examples if e["relpath"] not in taken]
            rng.shuffle(rest)
            chosen.extend(rest[: n - len(chosen)])
        chosen = sorted(chosen, key=lambda x: x["relpath"])
        LOG.info("Selected balanced eval subset: %d images (%s)", len(chosen),
                 pd.Series([e["ground_truth"] for e in chosen]).value_counts().to_dict())

    write_json(
        {
            "n": len(chosen),
            "seed": seed,
            "relpaths": [e["relpath"] for e in chosen],
            "class_distribution": pd.Series(
                [e["ground_truth"] for e in chosen]
            ).value_counts().to_dict(),
            "note": "Frozen evaluation subset. Reused verbatim by the fine-tuned "
                    "evaluation so the comparison is like-for-like.",
        },
        path,
    )
    return chosen


def run_inference(
    loaded: Any,
    examples: list[dict[str, Any]],
    prompt_ids: list[str],
    max_new_tokens: int = 128,
    do_sample: bool = False,
    results_csv: str | Path = "results/baseline/baseline_results.csv",
    raw_jsonl: str | Path = "results/baseline/baseline_raw_outputs.jsonl",
    tag: str = "base",
    limit: int | None = None,
) -> pd.DataFrame:
    """Query the model on every (image, prompt) pair and persist everything.

    Raw responses are streamed to JSONL as they are produced, so a crash or an
    interrupted Colab session never loses completed work.
    """
    from .evaluate import error_type
    from .inference import generate

    subset = examples[:limit] if limit else examples
    raw_path = resolve_path(raw_jsonl)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(subset) * len(prompt_ids)
    done = 0
    started = time.time()

    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for ex in subset:
            for pid in prompt_ids:
                done += 1
                try:
                    raw = generate(loaded, ex["image"], prompt_id=pid,
                                   max_new_tokens=max_new_tokens, do_sample=do_sample)
                    err = None
                except Exception as exc:  # noqa: BLE001 - record, never crash the run
                    raw, err = "", f"{type(exc).__name__}: {exc}"
                    LOG.error("Generation failed for %s / %s: %s", ex["relpath"], pid, err)

                parsed = parse_prediction(raw)
                row = {
                    "image_id": ex["image_id"],
                    "image_path": ex["image"],
                    "relpath": ex["relpath"],
                    "ground_truth": ex["ground_truth"],
                    "true_defect_type": ex.get("defect_type"),
                    "prompt_id": pid,
                    "raw_response": raw,
                    "parsed_prediction": parsed["prediction"],
                    "predicted_defect_type": parsed["defect_type"],
                    "confidence": parsed["confidence"],
                    # Track B: free text mapped onto the 12 dataset labels.
                    "predicted_class": canonicalize_defect_type(
                        parsed["defect_type"], parsed["prediction"]),
                    "format_ok": parsed["format_ok"],
                    "evidence": parsed["evidence"],
                    "generation_error": err,
                    "model_tag": tag,
                }
                row["correct"] = row["ground_truth"] == row["parsed_prediction"]
                row["error_type"] = error_type(row["ground_truth"], row["parsed_prediction"])
                rows.append(row)

                raw_fh.write(json.dumps(
                    {**row, "prompt_text": get_prompt(pid)}, ensure_ascii=False) + "\n")
                raw_fh.flush()

                if done % 10 == 0 or done == total:
                    rate = done / max(time.time() - started, 1e-6)
                    LOG.info("[%s] %d/%d (%.2f img-prompt/s)", tag, done, total, rate)

    df = pd.DataFrame(rows)
    out = resolve_path(results_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    LOG.info("Wrote %d rows to %s and raw outputs to %s", len(df), out, raw_path)
    return df


BALANCED_SUBSET_FILE = "results/baseline/balanced_subset.json"


def build_balanced_subset(
    examples: list[dict[str, Any]],
    seed: int = 42,
    subset_path: str | Path = BALANCED_SUBSET_FILE,
    reuse: bool = True,
) -> dict[str, Any]:
    """Select a class-balanced slice of the held-out test set for Track A.

    The OK class is the binding constraint, so the slice takes *every* OK image
    in the test split and an equal number of Defective images, sampled to
    spread across defect types and source groups. The held-out test split
    itself is never modified — this is a reporting view over it.
    """
    import random
    from collections import defaultdict

    path = resolve_path(subset_path)
    by_relpath = {e["relpath"]: e for e in examples}

    if reuse and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        missing = [r for r in saved["relpaths"] if r not in by_relpath]
        if missing:
            raise RuntimeError(
                f"{len(missing)} images from the saved balanced subset are absent "
                "from the current test split; the split changed."
            )
        LOG.info("Reusing balanced subset of %d images", len(saved["relpaths"]))
        return saved

    rng = random.Random(seed)
    ok = sorted([e for e in examples if e["ground_truth"] == "OK"],
                key=lambda x: x["relpath"])
    defective = [e for e in examples if e["ground_truth"] == "Defective"]

    # Spread the Defective side across defect types: round-robin over types,
    # so a 12-image sample is not dominated by whichever type is most frequent.
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in defective:
        by_type[str(e.get("defect_type"))].append(e)
    for items in by_type.values():
        items.sort(key=lambda x: x["relpath"])
        rng.shuffle(items)

    chosen_def: list[dict[str, Any]] = []
    types = sorted(by_type)
    rng.shuffle(types)
    while len(chosen_def) < len(ok):
        progressed = False
        for t in types:
            if by_type[t] and len(chosen_def) < len(ok):
                chosen_def.append(by_type[t].pop())
                progressed = True
        if not progressed:
            break

    chosen = sorted(ok + chosen_def, key=lambda x: x["relpath"])
    payload = {
        "n": len(chosen),
        "n_ok": len(ok),
        "n_defective": len(chosen_def),
        "seed": seed,
        "selection_procedure": (
            "Every OK image in the held-out test split was taken (the OK class is "
            "the binding constraint at 12 images), plus an equal number of "
            "Defective images sampled round-robin across defect types with "
            f"seed {seed}. The held-out test split was NOT modified; this is a "
            "balanced reporting view over it."
        ),
        "defect_type_counts": dict(
            pd.Series([e["defect_type"] for e in chosen]).value_counts().sort_index()
        ),
        "source_groups": sorted({str(e.get("group_key")) for e in chosen}),
        "n_source_groups": len({str(e.get("group_key")) for e in chosen}),
        "relpaths": [e["relpath"] for e in chosen],
        "note": "Track A (binary) primary reporting slice. Frozen; reused by the "
                "fine-tuned evaluation so the comparison is like-for-like.",
    }
    write_json(payload, path)
    LOG.info("Balanced Track-A subset: %d images (%d OK / %d Defective) across %d groups",
             payload["n"], payload["n_ok"], payload["n_defective"],
             payload["n_source_groups"])
    return payload
