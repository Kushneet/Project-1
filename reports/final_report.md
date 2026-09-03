# Casting Defect Detection with a Fine-Tuned Vision-Language Model

**Author:** Kushneet Kaur
**Date:** 2026-08-29
**Status:** Complete. Baseline, fine-tuning, fine-tuned evaluation and comparison
all executed; §19 (generalization on real images) remains out of scope.

> **No numerical result in this report is written by hand.** Each is produced by
> the corresponding script and copied from its output file under `results/`.

---

## 1. Abstract

This study measures how well a pretrained open-source vision-language model
(`Qwen/Qwen3-VL-4B-Instruct`) classifies industrial casting images as OK or
Defective without any task-specific training, then measures how much LoRA
fine-tuning on the target dataset changes that performance, holding the
evaluation images, prompts and decoding settings fixed.

On 178 frozen held-out images under three fixed prompts (534 generations per
run), the base model reached 0.7361 accuracy on the class-balanced view of the
binary task and 0.1220 accuracy on 12-class defect typing. After LoRA
fine-tuning on 838 examples (33.0M trainable parameters, 1.35% of the model,
74 min on one Tesla T4), 12-class accuracy rose to 0.8333 and macro F1 from
0.0371 to 0.8144 — the study's principal result. Binary classification did not
improve: accuracy on the imbalanced full test set rose from 0.6330 to 0.9176,
but on the balanced view it *fell* from 0.7361 to 0.5833 and macro F1 from
0.7361 to 0.4958, because the fine-tuned model learned to answer "Defective"
almost unconditionally. The cause is a 91.5% Defective training split, and every
one of the 24 regressions is an OK casting. Fine-tuning also removed the base
model's prompt sensitivity (accuracy spread across prompts 0.3596 → 0.0225).
All defect labels in this dataset are synthetic, so these results describe
recognition of generated defect textures, not real industrial defects.

## 2. Introduction

Casting defect inspection is predominantly manual: slow, costly, and limited by
human consistency. Vision-language models offer classification together with a
natural-language rationale, which is attractive for inspection work where an
operator wants to know *why* a part was flagged. Whether a general-purpose VLM
can do this out of the box, and how much task-specific adaptation helps, is an
empirical question — which is what this project measures.

## 3. Problem statement

Given a photograph of a cast metal component, determine whether it is
defect-free (OK) or defective, and where the labels support it, identify the
defect type — quantifying the contribution of fine-tuning against a properly
measured pretrained baseline.

## 4. Objectives

1. Select a practical open-source VLM that fine-tunes on a single GPU.
2. Query the base model on casting images **without training** and document it.
3. Analyse the dataset and establish what its labels actually support.
4. Prepare leakage-safe multimodal instruction data.
5. Fine-tune with LoRA/PEFT.
6. Evaluate on the identical held-out images with identical prompts.
7. Compare base against fine-tuned, quantitatively and qualitatively.
8. Test generalization to previously unseen images.
9. Deliver an upload-and-predict demo.

## 5. Dataset description

`simmoshaikh/casting-defect-detection` (Kaggle) — 1,493 images, 144.4 MB,
CC BY-SA 4.0, by Simran Shaikh. Twelve class folders: `ok` plus `cold_shut`,
`crack`, `dent`, `flash`, `inclusion`, `mixed_defects`, `pinhole`, `porosity`,
`scratch`, `shrinkage`, `surface_roughness`.

**Critical property.** The dataset card states the 11 defect classes were
*"synthesised directly onto real OK casting images using texture and geometry
transforms"*, followed by GrabCut-composited hand overlays. The defect labels
are therefore labels of **generated** defects. This constrains every claim in
this report and is treated as Limitation L1 throughout.

## 6. Dataset analysis

From `results/dataset_analysis/dataset_summary.json`, over the twelve labelled
class folders under `data/raw/output/defect_dataset` (the sibling
`hand_overlays/` and `visualizations/` directories are overlay assets and
rendered plots, not dataset samples, and are excluded).

| Property | Value |
|---|---|
| Files scanned / valid / corrupted | 1,200 / 1,200 / 0 |
| Classes | 12 (`ok` + 11 defect types) |
| Images per class | 100 for every class |
| Class percentages | 8.333% each — perfectly uniform |
| Image dimensions | 512 × 512 for all 1,200 (one distinct size, std 0) |
| Channels | 3 (RGB) for all |
| Formats | JPEG for all |
| Duplicate filenames | 0 |
| Exact duplicate groups | 0 |
| Cross-class duplicates | 0 |
| Total size on disk | 120.76 MB |

**The uniformity matters.** At the class level the dataset is perfectly balanced.
The imbalance that dominates the binary task is derived: eleven of the twelve
folders are defect classes, so collapsing them to OK/Defective yields an 11:1
ratio. This distinction is the root of Limitation L2 and of the central finding
in §17.

Plots: `class_distribution.png`, `image_dimensions.png`, `sample_images.png`.

## 7. Data preprocessing

Images are converted to RGB (the vision tower expects three channels) and the
vision-token budget is capped (`min_pixels`/`max_pixels`) to bound memory.

**Splitting.** Because defects were painted onto OK castings, one source
casting yields many images. A plain stratified split would leak a casting
between train and test, so the split is by **group** (inferred source casting).
`verify_no_leakage()` raises on any group or image hash spanning two splits.
Byte-identical duplicates are removed; duplicates with contradictory labels are
dropped entirely.

| Field | Value |
|---|---|
| Seed | 42 |
| Ratios | 0.70 / 0.15 / 0.15 |
| Strategy | Group split by source casting, class-balanced greedily |
| Counts | 838 train / 184 validation / 178 test |
| Binary label balance (train) | 767 Defective / 71 OK (91.5% Defective) |
| Binary label balance (validation) | 167 Defective / 17 OK |
| Binary label balance (test) | 166 Defective / 12 OK |

## 8. Base VLM selection

`Qwen/Qwen3-VL-4B-Instruct` — Apache-2.0, ungated, ~4B parameters, loaded via
`Qwen3VLForConditionalGeneration` + `AutoProcessor`. Selected for permissive
licensing, a current documented API, PEFT compatibility and Colab feasibility.
Rejected alternatives (Gemma-3-4B — gated; Qwen2.5-VL-3B — superseded, research
licence; Qwen3-VL-8B — too large for a free T4; LLaVA-1.5/1.6 — dated;
SmolVLM2 — weaker at fine detail) are documented in `PROJECT_DECISIONS.md` §2.

## 9. Baseline methodology

The pretrained model is loaded with **no adapter** and queried with three fixed
prompts over a class-balanced subset of the held-out test split. Decoding is
greedy at seed 42. The subset is written to
`results/baseline/eval_subset.json` and reused verbatim by the fine-tuned
evaluation. Prompts deliberately omit the dataset's class names so the base
model is not handed the answer space.

## 10. Baseline results

Run 2026-08-28 on a Tesla T4: 534/534 generations, 0 generation errors, 0
unparseable outputs, 43.6 min. Full detail in `reports/baseline_report.md`.

**Track A — binary**, primary balanced view (12 OK + 12 Defective images):

| Metric | Value |
|---|---|
| Accuracy / balanced accuracy | 0.7361 |
| Precision (Defective) | 0.7429 |
| Recall (Defective) | 0.7222 |
| F1 (Defective) | 0.7324 |
| Macro F1 | 0.7361 |
| OK recall | 0.7500 |
| Confusion matrix | TN 27, FP 9, FN 10, TP 26 |

Wilson 95% CI on OK recall is [0.468, 0.911], computed over 12 distinct OK
images rather than 36 (image, prompt) rows, since repeated prompts on one image
are not independent samples. The interval is wide; this number is indicative.

On the full test set the same run gives accuracy 0.6330, balanced accuracy
0.6873, macro F1 0.4882 — lower, because 187 of 498 Defective rows were missed
while the 12 OK images stayed the same.

**Track B — 12-class defect type**, full test set: coverage 0.4607 (246 of 534
responses mapped to a class), accuracy 0.1220 on those, macro F1 0.0371. Against
a 12-class chance level of 0.083, the base model is at chance.

**Prompt sensitivity.** Full-set accuracy by prompt: `prompt_1` 0.8427,
`prompt_2` 0.4831, `prompt_3` 0.5730 — a 36-point spread for three promptings of
the same question. On the balanced slice the spread is 8 points (0.7917 / 0.7083
/ 0.7083), which shows the full-set spread is largely the imbalanced set
rewarding whichever prompt makes the model say "Defective" more often.

## 11. Baseline error analysis

From `results/baseline/error_analysis.csv`, which records the misclassified rows.
Errors are categorised as missed defect (false negative), false alarm (false
positive), unparseable output, hallucinated defect type, confidently wrong, and
low confidence.

The base model produced **196 error rows out of 534**:

| Failure flag | Count |
|---|---|
| `none` (misclassified, no specific flag) | 97 |
| `confidently_wrong` | 95 |
| `low_confidence` | 3 |
| `hallucinated_defect_type` | 1 |

The dominant failure is the missed defect: 187 false negatives against 9 false
positives. The base model is conservative — it rarely calls a good part bad, and
frequently calls a bad part good. Ninety-five of its errors were stated with high
confidence, which is Limitation L3 in practice.

## 12. Fine-tuning methodology

LoRA (PEFT) applied to the language-model projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
vision tower frozen. Loss is computed on answer tokens only: prompt tokens,
padding and image placeholders are masked to `-100`. Targets are derived
strictly from ground-truth labels — no label is invented.

A sanity check (Phase 9) runs first: 8 examples, 2 steps, verifying data
loading, label masking, forward/backward passes, non-zero trainable parameters,
finite loss, and checkpoint save/reload.

## 13. Training configuration

From `config/config.yaml` (actual values used are recorded in
`results/training/logs/training_summary.json`):

| Parameter | Value |
|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Batch size × grad accumulation | 1 × 8 (effective 8) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Warmup ratio / weight decay | 0.03 / 0.01 |
| Precision | bf16 (fp16 on T4) |
| Gradient checkpointing | enabled |
| Quantization | 4-bit NF4 (CUDA only) |
| Seed | 42 |

## 14. Fine-tuned model

From `results/training/logs/training_summary.json`. Trained 2026-08-28 on a
Tesla T4 in 73.7 minutes.

| Field | Value |
|---|---|
| Trainable parameters | 33,030,144 of 2,448,667,136 (1.3489%) |
| Frozen vision-tower tensors | 315 |
| Optimizer steps | 315 (3 epochs, effective batch 8) |
| Final training loss | 0.0116 (step 310) |
| Final validation loss | 0.026514 (step 315) |
| Mean training loss over the run | 0.1985 |
| Adapter size on disk | 126 MB |

| Step | Training loss | Validation loss |
|---|---|---|
| 100 | 0.038318 | 0.037284 |
| 200 | 0.030465 | 0.027587 |
| 300 | 0.011489 | 0.026776 |
| 315 | 0.011576 | 0.026514 |

Validation loss fell monotonically and never rose, so no early stopping was
triggered. By epoch 3, however, training loss (0.0116) sits well below
validation loss (0.0265) — mild overfitting, and the reason no further epochs
were added. Curves: `results/training/training_curves.png`.

Two compatibility fixes were required to run the project's training code on the
Transformers 5.x that Colab installs, both recorded in `src/train.py`:
`warmup_ratio` and `logging_dir` were removed from `TrainingArguments` (the
configured 3% warmup is converted to an explicit 9 warmup steps of 312, leaving
the schedule unchanged), and the collator was extended to pass
`mm_token_type_ids`, which Qwen3-VL now requires to compute multimodal RoPE.
Neither fix alters the model, the data, the prompts or the evaluation settings.

## 15. Evaluation methodology

Identical images, identical prompts, identical decoding. `compare_models.py`
refuses to produce a comparison unless both result sets cover exactly the same
(image, prompt) pairs. Unparseable outputs count as errors by default; the
alternative treatment is also reported.

## 16. Fine-tuned results

Run 2026-08-29 on a Tesla T4: 534/534 generations, 0 generation errors, 0
unparseable outputs, 20.5 min. Nine settings were asserted identical to the
baseline's `run_metadata.json` before the results were accepted — model id, seed,
prompt ids, prompt SHA-256 prefixes, dtype, `min_pixels`, `max_pixels`, image
count, generation count — plus the full decoding configuration. All passed. Full
detail in `reports/evaluation_report.md`.

**Track A — binary**, primary balanced view:

| Metric | Value |
|---|---|
| Accuracy / balanced accuracy | 0.5833 / 0.5834 |
| Precision (Defective) | 0.5455 |
| Recall (Defective) | 1.0000 |
| F1 (Defective) | 0.7059 |
| Macro F1 | 0.4958 |
| OK recall | 0.1667 |
| Confusion matrix | TN 6, FP 30, FN 0, TP 36 |

On the full test set: accuracy 0.9176, balanced accuracy 0.5693, macro F1 0.5854,
confusion matrix TN 6, FP 30, FN 14, TP 484.

**Track B — 12-class defect type**, full test set: coverage 0.6742 (360 of 534),
accuracy 0.8333 on covered responses, macro precision 0.8073, macro recall
0.8338, macro F1 0.8144, weighted F1 0.8255.

Coverage is below 1.0 because the prompts deliberately never listed the twelve
class names, so metrics cover a self-selected and optimistically biased subset.
Coverage itself is the unbiased measure of how often a usable class name was
produced at all, and it rose by 21 points.

## 17. Base vs fine-tuned comparison

Over 534 matched (image, prompt) rows. Full detail in
`reports/comparison_report.md`.

**Track B — the principal result.**

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Coverage | 0.4607 | 0.6742 | +0.2135 |
| Accuracy | 0.1220 | 0.8333 | **+0.7113** |
| Macro F1 | 0.0371 | 0.8144 | **+0.7773** |
| Weighted F1 | 0.0299 | 0.8255 | +0.7956 |

Twelve-class defect typing moved from chance to usable. Track B is scored over
twelve uniformly represented classes, so this gain is not an artefact of label
imbalance.

**Track A — the result that must be reported twice.**

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Accuracy, full test set (166 Def / 12 OK images) | 0.6330 | 0.9176 | +0.2846 |
| Accuracy, balanced slice (12 Def / 12 OK images) | 0.7361 | 0.5833 | **−0.1528** |
| Macro F1, balanced slice | 0.7361 | 0.4958 | **−0.2403** |
| Balanced accuracy, full test set | 0.6873 | 0.5693 | −0.1180 |
| Recall (Defective) | 0.6245 | 0.9719 | +0.3474 |
| OK recall | 0.7500 | 0.1667 | **−0.5833** |
| False negatives | 187 | 14 | −173 |
| False positives | 9 | 30 | +21 |

The two accuracy rows are the same behavioural change viewed from two angles.
The fine-tuned model answers "Defective" almost unconditionally. On a test set
that is 93% Defective this is rewarded; on a balanced one it is not. **The
+0.2846 figure must not be quoted without the −0.1528 beside it.**

**Prompt sensitivity — a clean, separable gain.** Full-set accuracy spread across
the three prompts fell from 0.3596 (0.8427 / 0.4831 / 0.5730) to 0.0225 (0.9101 /
0.9326 / 0.9101).

**Efficiency.** Mean response length fell from 372 to 97 characters and
generation ran ~3x faster (0.56 vs 0.20 image-prompt/s).

## 18. Qualitative analysis

From `results/comparison/qualitative_comparison.csv` and
`results/comparison/comparison_summary.json`, over 534 rows:

| Quadrant | Count | Composition |
|---|---|---|
| `both_correct` | 314 | — |
| `fixed_by_finetuning` | 176 | 173 Defective, 3 OK |
| `broken_by_finetuning` | 24 | **24 OK, 0 Defective** |
| `both_wrong` | 20 | — |

The composition of the two changed quadrants is the finding. Every gain is
essentially a previously missed defect now caught; every single regression is an
OK casting newly called defective. Nothing correctly identified as Defective was
broken.

**The regressions are not simple guessing.** On `ok/ok_00000.jpg` the base model
answered *"Defect present: No"*; the fine-tuned model answered *"Classification:
Defective / Defect type: crack / Evidence: A crack is visible on the right side
of the part."*, and under `prompt_3` added a bounding box, `[0.85, 0.45, 0.95,
0.55]`, for a defect that does not exist. Fabricating a specific type and
location on a defect-free part is a more consequential failure than an
unconfident wrong label, and it is the reason `hallucinated_defect_type` rose
from 1 case to 16.

**Where both models still fail.** Of the 20 `both_wrong` rows, the examined cases
are `flash` defects — thin material at the part edge, the subtlest synthetic
texture in the dataset. On `flash/flash_00046.jpg` the fine-tuned model answered
*"Classification: OK / Reason: None / Confidence: 100%"* — a wrong answer at
stated full confidence, illustrating L3 directly.

**Character of errors changed, not just the count.** Base: 196 error rows, of
which 95 `confidently_wrong`. Fine-tuned: 44 error rows, of which 16
`confidently_wrong` and 16 `hallucinated_defect_type`. The base model's dominant
error was missing defects; the fine-tuned model's is inventing them.

## 19. Generalization testing

**Status: PENDING.** Infrastructure exists (`scripts/test_new_images.py`,
`data/external/`), but no genuinely new casting images with verified labels
have been added. Until they are, no claim is made about real-world
generalization.

Without verified labels, results from this script are **qualitative only** and
no accuracy is computed — enforced in code, not merely by convention.

## 20. Limitations

- **L1 — Synthetic defect labels.** The central limitation. Defect types were
  generated programmatically; results describe synthetic defect textures.
- **L2 — Class imbalance.** 11 defect folders vs 1 OK folder. Accuracy alone is
  misleading; macro-F1 and confusion matrices are reported.
- **L3 — Uncalibrated confidence.** Model-reported only; no calibration.
- **L4 — Generalization untested** (§19).
- **L5 — Single seed.** No confidence intervals.
- **L6 — No CNN reference.** A ResNet/EfficientNet baseline would likely
  outperform the VLM here; its absence is a gap, not evidence of VLM superiority.
- **L7 — Not certified.** Research prototype, not an industrial QC system.

## 21. Future work

Validate on real casting photographs with verified labels; add a CNN baseline
as the accuracy reference point; calibrate confidence; repeat across seeds;
extend from image-level classification to defect localisation.

## 22. Conclusion

LoRA fine-tuning of a 4B vision-language model on 838 examples — 33.0M trainable
parameters, 74 minutes on one free Tesla T4 — taught the model a twelve-class
defect vocabulary it did not previously possess. Track B accuracy rose from 0.1220
to 0.8333 and macro F1 from 0.0371 to 0.8144, from chance to usable. It also made
the model's behaviour stable: sensitivity to prompt wording, a 36-point accuracy
spread in the base model, fell to 2 points, and output format compliance became
exact.

It did not make the model better at the binary decision. Measured on the
class-balanced view that this project designated primary before any result was
seen, accuracy fell from 0.7361 to 0.5833 and macro F1 from 0.7361 to 0.4958. The
model learned to answer "Defective" almost unconditionally: it now finds every
defect in the balanced slice and correctly identifies only 6 of 36 OK rows. On the
full held-out set, which is 93% Defective, that same behaviour reads as a 28-point
accuracy gain. Both numbers are reported here because either alone is misleading.

The cause is identifiable and correctable. The training split is 91.5% Defective,
inherited from a dataset whose twelve uniformly balanced class folders collapse to
an 11:1 binary ratio. No class-balanced sampling or loss weighting was applied.
All 24 regressions being OK castings is the signature of a learned base rate
rather than a learned feature.

Three constraints bound every claim above. The defect labels are synthetic (L1),
so this measures recognition of generated textures and not of real casting
defects. There is no CNN reference point (L6), so nothing here shows a VLM is the
right tool for this task — a ResNet would likely be more accurate and far cheaper.
And the study uses a single seed on 178 images, 12 of them OK (L5), so the OK-side
numbers in particular rest on a small sample.

What the project does establish, within those bounds, is a properly controlled
before-and-after measurement: identical images, identical prompts, identical seed
and decoding, verified by assertion rather than by convention, with the
regressions reported alongside the gains.

## 23. References

1. Qwen Team. *Qwen3-VL*. Hugging Face. https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
2. Hugging Face. *Transformers documentation — Qwen3-VL*. https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
3. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
4. Hugging Face. *PEFT: Parameter-Efficient Fine-Tuning*. https://huggingface.co/docs/peft
5. Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
6. Shaikh, S. *Casting Defect Detection* (Kaggle dataset, CC BY-SA 4.0). https://www.kaggle.com/datasets/simmoshaikh/casting-defect-detection
7. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12 (2011).
