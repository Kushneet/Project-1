#!/usr/bin/env python3
"""Phase 12 — run the fine-tuned model on new, previously unseen images.

    python scripts/test_new_images.py --image path/to/image.jpg
    python scripts/test_new_images.py --dir data/external
    python scripts/test_new_images.py --dir data/external --labels labels.csv

Without verified labels this is a QUALITATIVE check only: predictions are
printed and saved, but no accuracy is computed. Supply --labels (a CSV with
``filename,label`` where label is OK/Defective) to get quantitative metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data_analysis import IMAGE_EXTS  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, write_json  # noqa: E402

LOG = get_logger("test_new_images")


def collect_images(image: str | None, directory: str | None) -> list[Path]:
    """Gather the image paths to test from --image and/or --dir."""
    paths: list[Path] = []
    if image:
        p = Path(image).expanduser()
        if not p.exists():
            raise SystemExit(f"Image not found: {p}")
        paths.append(p)
    if directory:
        d = resolve_path(directory)
        if not d.exists():
            raise SystemExit(f"Directory not found: {d}")
        paths.extend(sorted(
            p for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="A single image to test")
    parser.add_argument("--dir", default=None, help="A directory of images to test")
    parser.add_argument("--labels", default=None,
                        help="CSV with filename,label for verified ground truth")
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--base-model", action="store_true",
                        help="Use the base model instead of the fine-tuned one")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="results/new_images")
    args = parser.parse_args()

    if not args.image and not args.dir:
        raise SystemExit("Provide --image and/or --dir")

    cfg = load_config(args.config)
    prompt_id = args.prompt_id or cfg["baseline"]["primary_prompt_id"]
    paths = collect_images(args.image, args.dir)
    if not paths:
        raise SystemExit(
            "No images found.\n"
            "Place genuinely new casting images in data/external/ and re-run.\n"
            "The external generalization experiment is PENDING until you do."
        )
    LOG.info("Testing %d image(s) with prompt %s", len(paths), prompt_id)

    from src.inference import load_model, predict_image  # late import: pulls in torch

    adapter = None if args.base_model else cfg["training"]["output_dir"]
    if adapter and not resolve_path(adapter).exists():
        raise SystemExit(
            f"No fine-tuned adapter at {resolve_path(adapter)}.\n"
            "Run scripts/train_model.py first, or pass --base-model."
        )
    loaded = load_model(
        model_name=cfg["model"]["model_name"],
        adapter_path=adapter,
        dtype=cfg["model"]["dtype"],
        attn_implementation=cfg["model"]["attn_implementation"],
        min_pixels=cfg["model"].get("min_pixels"),
        max_pixels=cfg["model"].get("max_pixels"),
    )

    truth: dict[str, str] = {}
    if args.labels:
        lbl = pd.read_csv(resolve_path(args.labels))
        truth = dict(zip(lbl["filename"], lbl["label"]))
        LOG.info("Loaded %d verified labels", len(truth))

    results = []
    for p in paths:
        r = predict_image(p, loaded=loaded, prompt_id=prompt_id)
        r["filename"] = p.name
        r["ground_truth"] = truth.get(p.name)
        results.append(r)

        print("\n" + "-" * 56)
        print(f"Image      : {p.name}")
        print(f"Prediction : {r['classification']}")
        print(f"Defect Type: {r['defect_type']}")
        conf = r["confidence"]
        print(f"Confidence : {'n/a' if conf is None else f'{conf:.0f}%'} "
              f"(model-reported, not calibrated)")
        print(f"Evidence   : {r['evidence'] or '(none given)'}")
        if r["ground_truth"]:
            ok = r["ground_truth"] == r["classification"]
            print(f"Ground truth: {r['ground_truth']}  ->  {'CORRECT' if ok else 'WRONG'}")

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "new_image_predictions.csv", index=False)
    with (out / "new_image_raw_outputs.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = {
        "n_images": len(results),
        "model": cfg["model"]["model_name"],
        "adapter": adapter,
        "prompt_id": prompt_id,
        "prediction_counts": pd.Series(
            [r["classification"] for r in results]).value_counts().to_dict(),
    }
    labelled = [r for r in results if r["ground_truth"]]
    if labelled:
        n_correct = sum(r["ground_truth"] == r["classification"] for r in labelled)
        summary["quantitative"] = {
            "n_labelled": len(labelled),
            "n_correct": n_correct,
            "accuracy": round(n_correct / len(labelled), 4),
        }
    else:
        summary["quantitative"] = None
        summary["note"] = (
            "No verified labels supplied. These predictions are QUALITATIVE only "
            "and must not be reported as accuracy."
        )
    write_json(summary, out / "new_image_summary.json")

    print("\n" + "=" * 56)
    print(f"Tested {len(results)} image(s) -> {out}")
    if summary["quantitative"]:
        q = summary["quantitative"]
        print(f"Accuracy on {q['n_labelled']} labelled images: {q['accuracy']}")
    else:
        print("QUALITATIVE ONLY — no verified labels, so no accuracy is claimed.")
    print("=" * 56)


if __name__ == "__main__":
    main()
