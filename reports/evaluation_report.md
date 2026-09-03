# Evaluation Report — Fine-Tuned Model

> **Status: COMPLETE.** Run 2026-08-29 on a Tesla T4. Produced by the fine-tuned
> evaluation path (`scripts/evaluate_model.py` logic), 534/534 generations,
> 0 generation errors, 0 unparseable outputs, 20.5 min wall time.

## 1. Setup

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Adapter | `results/training/final_model/` |
| Fine-tuning | LoRA (PEFT), vision tower frozen |
| Evaluation subset | **Identical** to the baseline (`eval_subset.json`, `reuse=True`) |
| Prompts | **Identical** to the baseline |
| Decoding | Greedy, seed 42 |

Evaluation conditions were not altered between the two runs. `compare_models.py`
aborts if the two result sets do not cover the same (image, prompt) pairs.

Before any result was accepted, nine settings were asserted equal to the values
recorded in `results/baseline/run_metadata.json`: model id, seed, prompt ids,
prompt SHA-256 prefixes, dtype, `min_pixels`, `max_pixels`, image count, total
generations, and the full decoding configuration. All passed.

Inference used fp16 with **no quantization**, matching the baseline. The 4-bit
NF4 quantization used during training applies to training only.

## 2. Training configuration actually used

From `results/training/logs/training_summary.json`.

| Field | Value |
|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params | 33,030,144 of 2,448,667,136 (1.3489%) |
| Epochs / effective batch size | 3 / 8 (batch 1 × grad-accum 8) |
| Learning rate | 1e-4 |
| Optimizer steps | 315 |
| Final training loss | 0.0116 (step 310); 0.1985 averaged over the whole run |
| Final validation loss | 0.026514 (step 315) |
| Wall time | 73.7 min on a Tesla T4 |

Validation loss fell at every evaluation and never rose:

| Step | Training loss | Validation loss |
|---|---|---|
| 100 | 0.038318 | 0.037284 |
| 200 | 0.030465 | 0.027587 |
| 300 | 0.011489 | 0.026776 |
| 315 | 0.011576 | 0.026514 |

By epoch 3 training loss (0.0116) sits well below validation loss (0.0265). This
is mild overfitting. It did not warrant early stopping — validation loss was
still decreasing — but it is the reason more epochs were not added.

## 3. Metrics

Track A is binary OK vs Defective. Per `config/config.yaml`, the **balanced
slice is the primary reporting view**; the full test set is reported alongside
it because that set is 166 Defective against 12 OK images, which makes plain
accuracy on it easy to misread.

### Primary view — balanced slice (12 OK + 12 Defective images, 72 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.5833 |
| Balanced accuracy | 0.5834 |
| Precision (Defective) | 0.5455 |
| Recall (Defective) | 1.0000 |
| F1 (Defective) | 0.7059 |
| Macro F1 | 0.4958 |
| OK recall | 0.1667 |
| Unparseable outputs | 0 |
| False positives | 30 |
| False negatives | 0 |

### Secondary view — full held-out test set (178 images, 534 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.9176 |
| Balanced accuracy | 0.5693 |
| Precision (Defective) | 0.9416 |
| Recall (Defective) | 0.9719 |
| F1 (Defective) | 0.9565 |
| Macro F1 | 0.5854 |
| OK recall | 0.1667 |
| Unparseable outputs | 0 |
| False positives | 30 |
| False negatives | 14 |

### Confusion matrices

Balanced slice — `results/evaluation/finetuned_confusion_matrix_balanced.png`

```
                predicted OK   predicted Defective
true OK                    6                    30
true Defective             0                    36
```

Full test set — `results/evaluation/finetuned_confusion_matrix.png`

```
                predicted OK   predicted Defective
true OK                    6                    30
true Defective            14                   484
```

The two matrices share the same OK row: all 12 OK images are in both views.

**Reading these numbers.** The fine-tuned model finds every defective casting in
the balanced slice (Defective recall 1.0000) but classifies 30 of 36 OK rows as
Defective (OK recall 0.1667). It has acquired a strong prior toward "Defective".
On the full test set, where 93% of rows are Defective, that prior is rewarded and
accuracy reaches 0.9176. On the balanced slice, where the prior earns nothing, the
same behaviour yields 0.5833. Balanced accuracy — 0.5693 on the full set — is the
honest single number, and it is close to the 0.5 of a model that always answers
Defective.

The cause is in the training data, not in the training procedure:
`data/processed/train.jsonl` holds **767 Defective against 71 OK examples
(91.5% Defective)**. Eleven of the dataset's twelve class folders are defect
classes, so the derived binary label is imbalanced 11:1 even though the twelve
classes are perfectly balanced at 100 images each. The model learned the prior.
This is Limitation L2 made concrete.

## 4. Defect-type performance

The dataset provides 12 class folders, so defect-type supervision is enabled.
**These types are synthetic** (see PROJECT_DECISIONS §1.1) — any accuracy here
describes recognition of generated defect textures, not real defects.

Track B, full test set (`results/evaluation/finetuned_metrics.json`):

| Metric | Value |
|---|---|
| Coverage | 0.6742 (360 of 534 responses mapped to a class) |
| Accuracy (on covered responses) | 0.8333 |
| Macro precision | 0.8073 |
| Macro recall | 0.8338 |
| Macro F1 | 0.8144 |
| Weighted F1 | 0.8255 |

Confusion matrix: `results/evaluation/finetuned_defect_type_confusion_matrix.png`

Coverage is below 1.0 because the prompts never showed the model the 12-label
vocabulary — this was deliberate, so that the base model was not handed the answer
space. Metrics therefore cover a self-selected subset of responses and are
optimistically biased. Coverage itself is the honest measure of how often the
model produced a usable class name at all, and it rose from 0.4607 to 0.6742.

Against a 12-class chance level of 0.083, an accuracy of 0.8333 on covered
responses is a substantial result — with the standing caveat that the textures
being recognised are synthetic.

## 5. Error analysis

From `results/evaluation/error_analysis.csv`, which records only misclassified
rows. The fine-tuned model produced **44 error rows out of 534** against the base
model's 196.

| Failure flag | Fine-tuned | Base |
|---|---|---|
| `confidently_wrong` | 16 | 95 |
| `hallucinated_defect_type` | 16 | 1 |
| `low_confidence` | 0 | 3 |
| `none` (misclassified, no specific flag) | 12 | 97 |
| **Total error rows** | **44** | **196** |

Two things changed in character, not just in count:

- `confidently_wrong` fell from 95 to 16. The base model frequently asserted a
  wrong answer with high stated confidence; the fine-tuned model does so far less.
- `hallucinated_defect_type` rose from 1 to 16. Every one of these is an OK
  casting for which the fine-tuned model named a specific defect type. This is
  the same "Defective" prior visible in §3, expressed at the defect-type level.

## 6. Observations

1. **Defect-type recognition is where fine-tuning paid off.** Track B accuracy
   moved from near chance to 0.8333, and coverage rose by 21 points. The model
   learned both the label vocabulary and the synthetic textures.

2. **Binary classification became biased rather than better.** Defective recall
   is essentially perfect and OK recall collapsed to 0.1667. Reported on the
   imbalanced full test set this looks like a large improvement; reported on the
   balanced slice, which the configuration designates primary, it is a
   regression. Both views are given above deliberately.

3. **Prompt sensitivity effectively disappeared.** Base accuracy across the three
   prompts on the full test set ranged from 0.4831 to 0.8427 — a 36-point spread.
   Fine-tuned accuracy ranges from 0.9101 to 0.9326, a 2-point spread. Whatever
   else fine-tuning did, it made the model's behaviour stable across prompt
   wording.

4. **Output format was learned.** Mean response length fell from 372 to 97
   characters. The model now answers in the terse trained format instead of
   writing a paragraph of prose. A practical side effect: generation ran at
   ~0.56 image-prompt/s against the base model's ~0.20, roughly 3x faster.

5. **Zero unparseable outputs in both runs.** Parsing was never the bottleneck,
   so none of the differences above are artefacts of output formatting.

6. **The single-seed caveat applies** (L5). The OK class contributes only 12
   distinct images, so OK recall in particular rests on a small sample and should
   be treated as indicative.
