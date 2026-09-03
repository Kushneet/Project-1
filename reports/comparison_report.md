# Comparison Report — Base vs Fine-Tuned

> **Status: COMPLETE.** Run 2026-08-29 by `scripts/compare_models.py`, over
> 534 matched (image, prompt) rows.

Same images. Same prompts. Same decoding settings. The only variable is the
LoRA adapter.

## 1. Quantitative comparison

### Full held-out test set (178 images, 534 rows)

From `results/comparison/comparison.csv`.

| Metric | Base model | Fine-tuned | Δ |
|---|---|---|---|
| Accuracy | 0.6330 | 0.9176 | **+0.2846** |
| Precision (Defective) | 0.9719 | 0.9416 | −0.0303 |
| Recall (Defective) | 0.6245 | 0.9719 | **+0.3474** |
| F1 (Defective) | 0.7604 | 0.9565 | +0.1961 |
| Macro F1 | 0.4882 | 0.5854 | +0.0972 |
| False positives | 9 | 30 | +21 |
| False negatives | 187 | 14 | **−173** |
| Unparseable outputs | 0 | 0 | 0 |

Plot: `results/comparison/comparison_plot.png`

### Balanced slice (12 OK + 12 Defective images, 72 rows) — the primary view

`config/config.yaml` designates the balanced slice as the primary reporting view
for Track A. The picture there is the opposite.

| Metric | Base model | Fine-tuned | Δ |
|---|---|---|---|
| Accuracy | 0.7361 | 0.5833 | **−0.1528** |
| Balanced accuracy | 0.7361 | 0.5834 | −0.1527 |
| Precision (Defective) | 0.7429 | 0.5455 | −0.1974 |
| Recall (Defective) | 0.7222 | 1.0000 | +0.2778 |
| F1 (Defective) | 0.7324 | 0.7059 | −0.0265 |
| Macro F1 | 0.7361 | 0.4958 | **−0.2403** |
| OK recall | 0.7500 | 0.1667 | **−0.5833** |

Balanced accuracy on the full test set tells the same story as the balanced
slice: 0.6873 → 0.5693, a fall of 0.118.

**The two tables are not in conflict; they measure different things.** The full
test set is 93% Defective. A model that drifts toward answering "Defective"
gains on that set and loses nothing until it is asked to identify an OK part.
The +0.2846 accuracy gain and the −0.2403 macro-F1 loss are the same behavioural
change seen from two angles. Quoting the first without the second would
overstate what fine-tuning achieved.

### Per-prompt breakdown

Accuracy on the full test set, from `results/comparison/comparison.csv`:

| Prompt | Base | Fine-tuned | Δ |
|---|---|---|---|
| `prompt_1` | 0.8427 | 0.9101 | +0.0674 |
| `prompt_2` | 0.4831 | 0.9326 | +0.4495 |
| `prompt_3` | 0.5730 | 0.9101 | +0.3371 |
| **Spread (max − min)** | **0.3596** | **0.0225** | −0.3371 |

Macro F1 per prompt is less flattering, and on `prompt_1` it regresses:

| Prompt | Base macro F1 | Fine-tuned macro F1 | Δ |
|---|---|---|---|
| `prompt_1` | 0.6369 | 0.5762 | −0.0607 |
| `prompt_2` | 0.4008 | 0.5539 | +0.1531 |
| `prompt_3` | 0.4507 | 0.6124 | +0.1617 |

The base model was highly sensitive to prompt wording — a 36-point accuracy
spread across three prompts asking the same question. After fine-tuning that
spread is 2 points. This is a genuine and separable gain: whatever the model now
believes, it believes it consistently.

## 2. Qualitative comparison

Assessed from `results/comparison/qualitative_comparison.csv` (534 rows, all
four quadrants) and the two `error_analysis.csv` files.

| Dimension | Base model | Fine-tuned |
|---|---|---|
| Ability to detect defects | Misses many: 187 false negatives, Defective recall 0.6245 | Near-total: 14 false negatives, Defective recall 0.9719 |
| Consistency across prompts | Poor — accuracy spread 0.3596 across three prompts | Strong — spread 0.0225 |
| Hallucination | Rare at the type level (1 flagged case), because it rarely named a type at all | More frequent (16 cases), always naming a defect type on an OK casting |
| Confidence behaviour | 95 `confidently_wrong` rows; asserts wrong answers firmly | 16 `confidently_wrong` rows; still uncalibrated but far less often confidently wrong |
| Explanation quality | Long free prose, often describing the part rather than deciding (mean 372 chars) | Terse trained format, decision-first (mean 97 chars); less explanatory detail |
| Output-format compliance | Followed the prompt format loosely; 0 unparseable | Follows the trained format exactly; 0 unparseable |
| False positives | 9 — conservative, rarely cries wolf | 30 — flags most OK castings as defective |
| False negatives | 187 — the dominant failure mode | 14 |

The trade is visible in one line: the base model's dominant error was missing
defects; the fine-tuned model's dominant error is inventing them.

## 3. Representative examples

Every evaluated (image, prompt) row is labelled with one of four quadrants.
From `results/comparison/comparison_summary.json`, over 534 rows:

| Quadrant | Count | What it tells us |
|---|---|---|
| `both_correct` | 314 | Task was already within the base model's reach |
| `both_wrong` | 20 | Hard cases fine-tuning did not fix |
| `fixed_by_finetuning` | 176 | **The value added by fine-tuning** |
| `broken_by_finetuning` | 24 | **Regressions — must be discussed, not hidden** |

The composition of the two changed quadrants is the finding:

- `fixed_by_finetuning` — **173 Defective, 3 OK.** Almost every gain is a
  previously missed defect now caught.
- `broken_by_finetuning` — **24 OK, 0 Defective.** Every single regression is an
  OK casting newly misclassified as Defective. Nothing that was correctly called
  Defective was broken.

That asymmetry is the numerical signature of the "Defective" prior described in
`evaluation_report.md` §3.

### Examples where both were correct

`cold_shut/cold_shut_00009.jpg`, `prompt_1`, truth Defective:

- Base → Defective: *"A wavy, dashed line is superimposed across the casting, indicating a defect…"*
- Fine-tuned → Defective, type `cold_shut`: *"Classification: Defective / Defect type: cold_shut"*

Both reach the right binary answer, but only the fine-tuned model names the type.
This is typical of the quadrant: 314 rows where the base model already succeeded,
and fine-tuning added type-level precision on top.

### Examples where both were wrong

`flash/flash_00042.jpg`, `prompt_3`, truth Defective:

- Base → OK: *"The casting shows a clean, concentric circular structure with no visible cracks, porosity…"*
- Fine-tuned → OK: *"Classification: OK / Defect type: None"*

`flash/flash_00046.jpg`, `prompt_1`, truth Defective:

- Base → OK: *"…no visible cracks, porosity, or surface defects; the concentric r…"*
- Fine-tuned → OK: *"Classification: OK / Reason: None / Confidence: 100%"*

Both `both_wrong` examples above are `flash` defects — thin material at the part
edge, the subtlest of the synthetic textures. Note the fine-tuned model states
100% confidence on a wrong answer, which is L3 (uncalibrated confidence) in
practice.

### Base wrong → fine-tuned correct

`cold_shut/cold_shut_00004.jpg`, `prompt_2`, truth Defective:

- Base → OK: *"Defect present: No / Defect type: Unknown / Evidence: The image shows a metallic casting with concentric rings and…"*
- Fine-tuned → Defective, type `cold_shut`: *"A cold shut is visible as a wavy line across the su…"*

The same image under `prompt_3` shows the identical pattern. The base model
describes the part accurately and then declines to call it defective; the
fine-tuned model identifies the specific defect. This is the shape of most of
the 176 fixed rows.

### Base correct → fine-tuned wrong

`ok/ok_00000.jpg`, `prompt_2`, truth OK:

- Base → OK: *"Defect present: No / Defect type: Unknown / Evidence: The image shows a circular industrial casting with concentri…"*
- Fine-tuned → Defective, type `crack`: *"A crack is visible on the right side of the part."*

The same image under `prompt_3`:

- Base → OK: *"Prediction: OK / Evidence: …There ar[e no visible defects]"*
- Fine-tuned → Defective, type `crack`, with a bounding box: *"Defect location: [0.85, 0.45, 0.95, 0.55]"*

The fine-tuned model does not merely guess "Defective" — it fabricates a specific
defect type and, when the prompt asks for it, a plausible-looking bounding box,
on a casting with no defect. This is the most important failure mode in the study
and the reason the regression quadrant is reported rather than omitted.

## 4. Interpretation

**What fine-tuning demonstrably achieved.**

1. Defect-type recognition went from near-chance to usable: Track B accuracy
   0.1220 → 0.8333, macro F1 0.0371 → 0.8144, coverage 0.4607 → 0.6742. This is
   the strongest result in the study and it is not an artefact of class balance,
   because Track B is scored over 12 classes that are uniformly represented.
2. Prompt sensitivity was largely eliminated: accuracy spread across the three
   prompts fell from 0.3596 to 0.0225.
3. Output-format compliance became exact, and responses shortened from a mean of
   372 to 97 characters — which also made inference roughly 3x faster.
4. Missed defects fell sharply: 187 false negatives to 14.

**What fine-tuning did not achieve.**

Binary discrimination did not improve; it shifted. The model traded 173 recovered
defects for 21 new false alarms and the near-total loss of its ability to
recognise an OK casting (OK recall 0.7500 → 0.1667). On the primary balanced
view, macro F1 fell by 0.2403. A model that answers "Defective" almost always
scores well on a test set that is 93% Defective, and that is most of what the
+0.2846 full-set accuracy gain represents.

**Why.** `data/processed/train.jsonl` contains 767 Defective and 71 OK examples —
91.5% Defective. The dataset's twelve class folders are perfectly balanced at 100
images each, but eleven of the twelve are defect classes, so the derived binary
label inherits an 11:1 imbalance. The model learned the base rate. This is
Limitation L2, and it is a property of the experiment design rather than a bug in
the fine-tuning code.

**What would address it**, in order of expected effect: class-balanced sampling
or loss weighting during fine-tuning; oversampling the OK class or augmenting it;
adding a decision threshold calibrated on the validation split rather than taking
the model's argmax; and reporting balanced accuracy as the headline metric
throughout. None of these were attempted here, and each is a concrete piece of
future work rather than a speculative suggestion.

**Bottom line.** Fine-tuning a 4B vision-language model with LoRA on 838 examples
for 74 minutes on a single free T4 taught it a 12-class defect vocabulary it did
not previously have, and made its behaviour stable and well-formatted. It did not
teach it to tell a good casting from a bad one better than it already could — on
the balanced view it got worse at that — and the reason is traceable to an 11:1
label imbalance that the study design did not correct for.

## 5. Caveat

Any improvement measured here is improvement at detecting **synthetic** defect
textures on this dataset's images. It is not evidence of improved detection of
real casting defects. That requires the Phase-12 experiment on real, verified
images (currently pending).
