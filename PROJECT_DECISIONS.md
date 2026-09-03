# Project Decisions

Every decision here was made after inspecting the actual environment and the
live model/library documentation on **2026-08-26**. Where something has not yet
been verified against real data, it says so explicitly.

---

## 1. Dataset

**Selected:** [`simmoshaikh/casting-defect-detection`](https://www.kaggle.com/datasets/simmoshaikh/casting-defect-detection)
· v1 · 1,493 images · 144.4 MB · CC BY-SA 4.0 · creator: Simran Shaikh.

### 1.1 Labels — 12 classes, but they are SYNTHETIC

The dataset card documents 12 class folders: `ok`, `cold_shut`, `crack`,
`dent`, `flash`, `inclusion`, `mixed_defects`, `pinhole`, `porosity`,
`scratch`, `shrinkage`, `surface_roughness`.

Defect-type labels therefore **do exist**, and the project trains for them.
However, the dataset's own description states how they were produced:

> "11 defect types synthesised directly onto real OK casting images using
> texture and geometry transforms via `casting_defect_augmentation.py`."

and, in a second stage:

> "Real hand images composited onto ALL casting images using OpenCV GrabCut
> background removal."

**Consequence, stated plainly:** a model trained on this data learns to detect
*programmatically generated defect textures*, not physically real casting
defects. Any claim of the form "this model detects real cracks or porosity"
is **not supported** by this dataset. Every report in this repository must
carry that caveat. This is recorded as Limitation L1.

This was raised with the project owner before implementation. The decision was
to proceed with the assigned dataset only, with the limitation documented
rather than worked around.

### 1.2 Rejected supplementary dataset

`ravirajsinh45/real-life-industrial-dataset-of-casting-product` (7,348 real
photographs of submersible pump impellers, 300×300 grayscale, binary
`ok_front`/`def_front`, official train/test split) was evaluated as a real-image
benchmark and **not adopted** for the main experiment, by decision of the
project owner.

`scripts/download_dataset.py --which external` can still fetch it into
`data/external/` for the Phase-12 *qualitative* generalization check. It is
never trained on and never contributes to a headline metric. Note its licence
is CC BY-NC-ND 4.0 (non-commercial, no-derivatives) — acceptable for academic
evaluation, but it must not be redistributed or used commercially.

### 1.3 Measured dataset facts (verified 2026-08-26, post-download)

The archive arrived already extracted at `~/Downloads/output/`; there was no
ZIP to unpack. It was copied to `data/raw/output/` leaving the original intact.

| Property | Measured value |
|---|---|
| Labelled images | **1,200** (not 1,493 — see below) |
| Classes | 12, **exactly 100 images each** |
| Dimensions | 512x512, uniform across all 1,200 |
| Channels / mode | 3 (RGB), JPEG |
| Corrupted images | **0** |
| Duplicate filenames | 0 |
| Byte-identical duplicates | 0 |
| Official train/test split | **None** — flat class folders only |

**The "1,493 images" on the dataset card is not 1,493 labelled samples.**
The labelled set is 1,200 (`output/defect_dataset/`). The remainder is
`hand_overlays/` (276 unlabelled overlay renderings) and `visualizations/`
(16 rendered plots and sample grids). Neither is class-labelled, so both are
**excluded** from analysis, training and evaluation. `config.data.raw_dir`
points at `data/raw/output/defect_dataset` for this reason.

**Balance depends on the task.** The 12-way task is perfectly balanced
(100/class). The binary task is **heavily imbalanced: 100 OK vs 1,100
Defective (1:11)**. An "always Defective" classifier scores 91.7% accuracy, so
accuracy alone is meaningless here — macro-F1 and the confusion matrix govern.

### 1.4 Leakage: filenames do NOT encode the source casting

The original plan grouped by a source-casting id parsed from filenames. **On
the real data that heuristic is invalid**, and it was replaced.

Real filenames are per-class counters: `ok_00000.jpg`, `crack_00000.jpg`, ...
The index is a within-class counter, not a casting id. Measured:

- **Index-match rate 2.9%** (32/1100). For each defect image, the nearest OK
  image by normalised correlation is the same-index OK image only 2.9% of the
  time — chance level for 100 candidates. `ok_00042` and `crack_00042` are
  **unrelated images**.
- Mean similarity to the index-matched OK image is **0.28**, versus **0.72** to
  the best-matching OK image. There is no index correspondence.

**But source castings genuinely are reused across classes**, so the leakage
risk is real:

- **360 cross-class image pairs exceed 0.99** normalised correlation;
  2,570 exceed 0.90.
- Examples: `ok_00083` <-> `scratch_00060` (0.9995),
  `ok_00007` <-> `flash_00010` (0.9994), `ok_00014` <-> `pinhole_00000` (0.9991).
- At full resolution `ok_00007` vs `flash_00010` differ by >15 grey levels over
  only **1.35% of the image area** — a shared base casting with a small painted
  defect, exactly the predicted leakage mechanism.

**Decision: group by image content, not filename.** Images are embedded as
64x64 contrast-normalised thumbnails (per-image mean/std normalisation, so a
global exposure shift does not break matching), linked when normalised
correlation exceeds `split.similarity_threshold`, and each connected component
becomes one group. Implemented in `src/dataset.compute_content_groups`.

Result at threshold 0.95: **854 groups**, largest 28, 809 singletons, and
**45 multi-image groups covering 391 images (33% of the dataset)** — those 391
images are precisely the ones that would have leaked. Agreement between the
content grouping and the old filename heuristic is **0.0%**, confirming the
filename key was meaningless.

`split.group_by: filename` is retained for datasets whose filenames genuinely
encode the source casting.

### 1.5 Leakage verification on the real data

`verify_no_leakage()` **passes**: 0 groups and 0 content hashes span more than
one split.

An independent audit — computed from raw pixels without consulting group
labels — confirms it. The highest similarity across any split boundary is
**0.9491** (train<->test), below the 0.95 grouping threshold; zero cross-split
pairs reach 0.95.

Threshold sensitivity (0.85 to 0.99) was checked: the split stays leakage-free
at every value and split proportions stay stable. Below 0.88 the largest group
balloons (272 images at 0.85) from over-merging, so 0.95 sits in the stable
middle.

### 1.6 Actual splits

| Split | Images | Groups | OK | Defective |
|---|---|---|---|---|
| train | 838 (69.8%) | 586 | 71 | 767 |
| validation | 184 (15.3%) | 142 | 17 | 167 |
| test | 178 (14.8%) | 126 | **12** | 166 |

Seed 42, group split, class-balanced greedily within groups.

**Open issue (L8): the test split holds only 12 OK images.** That follows from
the dataset having just 100 OK images in total, and it caps the statistical
power of the binary evaluation — one misclassified OK image moves OK-recall by
~8 percentage points. This is flagged for a decision before the baseline runs.

---

## 1.7 Two evaluation tracks

Decided after the real-data analysis showed the dataset is balanced 12-way but
1:11 binary.

**TRACK A — PRIMARY: binary defect detection (OK vs Defective).**
Reported on a **class-balanced slice** of the held-out test split: all 12 OK
images plus 12 Defective images sampled round-robin across defect types with
seed 42, spanning 20 source groups. The held-out split is **not modified** —
the slice is a reporting view. Metrics on the full 178-image test set are
reported alongside it. Reported: accuracy, balanced accuracy, precision,
recall, F1, macro-F1, OK recall, Defective recall, false positives, false
negatives, confusion matrix, and Wilson 95% intervals.

**TRACK B — SECONDARY: 12-class defect-type classification.**
Uses all 12 original labels on the full 178-image test set. Reported: macro
precision/recall/F1, weighted F1, per-class precision/recall/F1, confusion
matrix, and **coverage** — the fraction of responses that yielded a
recognisable class.

The 12 original labels are preserved end to end. The binary label is *derived*
from them (`is_defective_label`) and never replaces them; every results row
carries both `true_defect_type` and `ground_truth`.

Because the prompts do not leak the class list, the base model emits free-form
descriptions. `canonicalize_defect_type()` maps those onto the 12 labels via a
synonym table ("blowhole" -> porosity, "sink mark" -> shrinkage); anything
unmatched is recorded as `Unmatched` and excluded from Track B metrics but
counted in coverage. Metrics on matched rows only are a self-selected subset
and therefore optimistically biased — stated in the output itself.

### Statistical handling of repeated prompts

Each image is queried with 3 prompts, so results rows are **not independent
samples**. Wilson 95% intervals and the small-sample warning are computed from
the number of *distinct images*, not rows: 12 OK images, not 36 rows. Without
this the interval on OK recall would be roughly 40% too narrow.

---

## 2. Model selection

| Field | Value |
|---|---|
| **Model name** | `Qwen/Qwen3-VL-4B-Instruct` |
| **Model version** | Qwen3-VL series, Instruct variant, dense |
| **Base language model** | Qwen3 4B |
| **Vision encoder** | Qwen3-VL ViT with DeepStack multi-level feature fusion |
| **Approximate parameters** | ~4B total |
| **Licence** | Apache-2.0 (verified via the HF API — ungated) |
| **Expected VRAM** | ~9 GB bf16 inference; ~7–10 GB LoRA training with 4-bit + grad checkpointing |
| **Fine-tuning method** | LoRA (PEFT), vision tower frozen |
| **Transformers classes** | `Qwen3VLForConditionalGeneration` + `AutoProcessor` |

### Why this model

1. **Ungated and Apache-2.0.** Verified live against the HF API: `gated: False`.
   A student can reproduce this without an approval queue or a token.
2. **Current, documented API.** The loading and chat-template code in
   `src/inference.py` follows the current Transformers `qwen3_vl` documentation,
   including `apply_chat_template(..., tokenize=True, return_dict=True)` and
   dropping `token_type_ids` before `generate()`.
3. **Fits Colab.** ~4B parameters runs on a free T4 with 4-bit quantization,
   gradient checkpointing and a capped vision-token budget.
4. **Real PEFT support.** Standard attention/MLP projection names, so LoRA
   attaches without custom surgery.
5. **Strong for its size** on document/visual understanding benchmarks, which
   matters because the task is fine-grained surface inspection.

### Rejected alternatives

| Model | Why rejected |
|---|---|
| **Gemma-3-4B-IT** | Gated on Hugging Face (`gated: manual`). Requires per-user approval — a poor reproducibility story for a submitted project. |
| **Qwen2.5-VL-3B-Instruct** | Superseded, and its licence is the Qwen Research licence rather than Apache-2.0. Qwen3-VL-4B is newer, better, and more permissively licensed. |
| **Qwen3-VL-8B / larger** | ~16 GB+ in bf16. Marginal on a free T4 and slow to fine-tune, for little gain on a binary/12-class task. |
| **LLaVA-1.5/1.6 family** | Older architecture; most tutorials target deprecated APIs. Weaker than current small VLMs at fine detail. |
| **SmolVLM2-2.2B** | Attractively small, but noticeably weaker at fine-grained visual discrimination, which is the whole task here. |
| **CNN baselines (ResNet/EfficientNet)** | Would likely *beat* a VLM on this task — but the assignment is explicitly a VLM before/after fine-tuning study. Noted in Future Work as the proper accuracy reference point. |

---

## 3. Compute

**Local machine:** MacBook Air M4, 16 GB unified memory, no NVIDIA GPU
(`torch.cuda.is_available() == False`, MPS only).

**Decision:** dataset analysis, splitting, data preparation and the test suite
run locally. **Baseline inference and fine-tuning run on Google Colab** (T4 or
L4). Reasons: `bitsandbytes` 4-bit quantization is CUDA-only, and MPS training
of a 4B VLM on 16 GB shared memory is not viable in a reasonable time.

This is a **compute constraint, not a methodology change**. The experimental
protocol — same images, same prompts, base measured before fine-tuned — is
identical wherever it runs.

---

## 4. Experimental protocol

The order mandated by the brief is enforced in code, not just documented:

1. `scripts/analyze_dataset.py` — measure the dataset.
2. `scripts/prepare_training_data.py` — group split, leakage check, JSONL.
3. `scripts/run_baseline.py` — **base model, no adapter**, freezes the
   evaluation subset to `results/baseline/eval_subset.json`.
4. `scripts/train_model.py --sanity-check` then `scripts/train_model.py`.
   **`train_model.py` refuses to start a full run if baseline results are
   absent**, so the order cannot be reversed by accident.
5. `scripts/evaluate_model.py` — reloads the *frozen* subset with
   `reuse=True` and the identical prompt list.
6. `scripts/compare_models.py` — **aborts** if the two runs do not cover
   exactly the same (image, prompt) pairs.

### Fixed decisions

- **Seed 42** everywhere (`project.seed`, `split.seed`, `inference.seed`).
- **Greedy decoding** (`do_sample: false`) so both runs are reproducible and
  differences are attributable to the weights, not sampling noise.
- **Three prompts**, frozen in `src/prompts.py`, used identically in Phases 4
  and 10. A test asserts they do not leak the dataset's class names, which
  would unfairly inflate the baseline.
- **Unparseable outputs count as errors** by default. Silently dropping them
  would flatter a base model that fails to follow the output format; both
  variants are reported.
- **Positive class = Defective**, so recall measures missed defects — the
  costly error in real inspection.

---

## 5. Known limitations (carried into every report)

- **L1 — Synthetic defects.** Defect types are programmatically generated.
  Results describe detection of synthetic defect textures, not real casting
  defects.
- **L2 — Class imbalance (measured).** 100 OK vs 1,100 Defective (1:11). An
  "always Defective" classifier scores 91.7% accuracy, so accuracy alone is
  meaningless; macro-F1 and the confusion matrix govern, and the evaluation
  subset is class-balanced by construction. The 12-way task is balanced.
- **L3 — Uncalibrated confidence.** Every confidence figure is the model's own
  self-reported number. It is labelled "model-reported" throughout and no
  calibration has been performed.
- **L4 — Generalization untested.** Until real, verified-label casting images
  are placed in `data/external/`, the Phase-12 experiment is **pending**, and
  no claim about real-world generalization is made.
- **L5 — Single seed.** Results come from one seed; no confidence intervals
  over repeated runs.
- **L8 — Thin OK test set.** Only 12 OK images reach the test split (the
  dataset holds just 100 OK images in total). Binary metrics on the OK class
  have wide error bars; per-class counts must be reported alongside rates.
