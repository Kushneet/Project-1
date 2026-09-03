# Running the baseline on Google Colab

The Colab run is the **authoritative baseline experiment**. The local Apple-MPS
attempt was an infrastructure test only and produced no evaluation results.

## Storage approach

Google Drive is nearly full, so the project is **never extracted into Drive**.
Drive holds one ZIP; the notebook reads it and unpacks into the Colab runtime
disk at `/content/`. **Nothing is written back to Drive.**

```
Google Drive                          Colab runtime disk (~78-107 GB free)
└── MyDrive/                          └── /content/
    └── casting-defect-vlm-colab.zip      ├── casting-defect-vlm/     <- extracted here
        (118 MB, read-only)               │   └── results/baseline/   <- results land here
                                          └── baseline_results.zip    <- downloaded to your PC
```

The runtime disk is **wiped when the session ends** — download the results ZIP
before closing Colab.

## 1. Upload the ZIP to Drive

Put `casting-defect-vlm-colab.zip` in the **root** of your Drive:

```
MyDrive/casting-defect-vlm-colab.zip
```

Do not unzip it in Drive. If you put it elsewhere, edit `ZIP_IN_DRIVE` in
section 2 of the notebook.

## 2. Open the notebook

`notebooks/02_base_model_baseline_colab.ipynb`

**Runtime -> Change runtime type -> T4 GPU.**

No path editing is needed — `PROJECT` is `/content/casting-defect-vlm`.

## 3. Run cells in order, and stop at the smoke test

| Section | What it does |
|---|---|
| 1 | GPU check |
| 2 | Mount Drive, storage check, extract to `/content`, verify structure |
| 3 | pip install |
| 4 | Print the frozen prompts + hashes |
| 5 | Load the base model (no adapter) |
| 6 | **Timed smoke test — STOP and read the projection** |
| 6b | Approval gate (`APPROVED_FOR_FULL_RUN = False`) |
| 7 | Full 534-generation run (blocked until you approve) |
| 8 | Validation asserts |
| 9-13 | Metrics, examples, report |
| 14 | Package + download results |
| 15 | Stop |

Section 2 prints:

```
Google Drive free space      : X.XX GB
Colab runtime free space     : XX.XX GB
Project ZIP size             : 123 MB (1304 entries)
Estimated extracted size     : 125 MB
Model cache needed           : 9.0 GB
Total required on /content   : 12.13 GB
```

It **stops with an error** if the runtime disk cannot fit that. Low *Drive*
space is fine — Drive is only read from.

Section 6b means a "Run all" cannot start the 534-generation run. Review the
projected runtime, then set `APPROVED_FOR_FULL_RUN = True` and continue.

## 4. Download the results

Section 14 zips **only** `results/baseline/` (a few MB) and downloads it.
If the browser blocks it, use the file browser (folder icon) ->
`/content/baseline_results.zip`.

Then locally:

```bash
cd ~/casting-defect-vlm
unzip -o ~/Downloads/baseline_results.zip -d results/baseline/
.venv/bin/python scripts/generate_baseline_report.py
```

## 5. Then stop

Do not start fine-tuning until the baseline is saved and verified.
`scripts/train_model.py` enforces this — it refuses a full training run when
`results/baseline/baseline_results.csv` is absent.

## If something breaks

Report the exact error. Do **not** work around it by changing the model, the
prompts, the evaluation set, the seed, or the decoding settings — any of those
silently breaks comparability with the fine-tuned run.

- **ZIP not found** — check it is at `MyDrive/casting-defect-vlm-colab.zip`, or
  edit `ZIP_IN_DRIVE`.
- **INSUFFICIENT COLAB DISK SPACE** — Runtime -> Disconnect and delete runtime,
  then start a fresh T4 session.
- **`Qwen3VLForConditionalGeneration` import fails** — `transformers` is too old.
  `!pip install -q git+https://github.com/huggingface/transformers`, restart the
  runtime, re-run from section 2 (the extract is idempotent).
- **CUDA OOM** — lower `model.max_pixels` in `config/config.yaml` (602112 ->
  401408) and record it in `PROJECT_DECISIONS.md`. It changes the vision-token
  budget, so the fine-tuned run must use the identical value.
