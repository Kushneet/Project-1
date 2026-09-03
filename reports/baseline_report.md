# Baseline Report — Base VLM, No Fine-Tuning

> **Status: COMPLETE.** Every number below was produced by `scripts/run_baseline.py` and read from `results/baseline/`. No value was written by hand.

> **The model was NOT fine-tuned.** No LoRA adapter, no training of any kind — the pretrained weights were queried directly.


## 1. Run metadata

| Field | Value |
|---|---|
| run | `baseline` |
| fine_tuned | `False` |
| adapter | `None` |
| model_id | `Qwen/Qwen3-VL-4B-Instruct` |
| model_class | `Qwen3VLForConditionalGeneration` |
| transformers_version | `5.15.1` |
| torch_version | `2.11.0+cu128` |
| python_version | `3.13.15` |
| platform | `Linux-6.6.122+-x86_64-with-glibc2.35` |
| device | `cuda:0` |
| dtype | `torch.float16` |
| attn_implementation | `sdpa` |
| generation | `{'do_sample': False, 'max_new_tokens': 128, 'decoding': 'greedy'}` |
| min_pixels | `200704` |
| max_pixels | `602112` |
| seed | `42` |
| prompt_ids | `['prompt_1', 'prompt_2', 'prompt_3']` |
| prompt_sha256_16 | `{'prompt_1': 'b2ad84caece22b78', 'prompt_2': '2c3e3f3333f50cbf', 'prompt_3': 'ec538eb4639255d2'}` |
| n_images | `178` |
| n_prompts | `3` |
| total_generations | `534` |
| balanced_slice_images | `24` |
| started_utc | `2026-08-28T07:06:39+00:00` |
| gpu_name | `Tesla T4` |
| gpu_memory_gb | `15.6` |
| finished_utc | `2026-08-28T07:50:14+00:00` |
| total_inference_seconds | `2614.68` |
| avg_generation_seconds | `4.896` |
| model_load_seconds | `178.21` |

## 2. Evaluation design

- **Full held-out test set:** 178 images x 3 prompts = **534 generations**.
- **Track A (primary)** reports on the frozen balanced slice: **12 OK + 12 Defective = 24 images**, seed 42, 20 source groups.
- **Track B (secondary)** uses the full test set for 12-class defect type.

> Subset selection: Every OK image in the held-out test split was taken (the OK class is the binding constraint at 12 images), plus an equal number of Defective images sampled round-robin across defect types with seed 42. The held-out test split was NOT modified; this is a balanced reporting view over it.


## 3. Prompts (frozen, verbatim)

<details><summary><code>prompt_1</code></summary>

```
You are an industrial casting quality inspection assistant.

Inspect the uploaded casting image carefully.

Determine whether the casting is:

1. OK
2. Defective

Return your answer in this format:

Classification: OK or Defective
Reason: brief visual explanation
Confidence: 0-100%
```
</details>

<details><summary><code>prompt_2</code></summary>

```
Inspect this industrial casting image for visible defects.

Determine whether a defect is present.

If the available dataset labels support a specific defect type, identify it.
If a specific defect type cannot be established, write Unknown.

Return:

Defect present: Yes or No
Defect type: [label or Unknown]
Evidence: brief explanation
Confidence: 0-100%
```
</details>

<details><summary><code>prompt_3</code></summary>

```
You are performing visual quality inspection of a manufactured casting.

Carefully inspect the image and classify it as OK or Defective.

Do not invent information that cannot be visually supported.

Return:

Prediction:
Evidence:
Confidence:
```
</details>

These are used **unchanged** for the fine-tuned evaluation. They do not list the dataset's class names, so the base model is not handed the answer space.


## 4. TRACK A — PRIMARY: binary OK vs Defective

### 4.1 Balanced reporting slice (12 OK + 12 Defective)

| Metric | Value |
|---|---|
| Accuracy | 0.7361 |
| Balanced accuracy | 0.7361 |
| Precision (Defective) | 0.7429 |
| Recall (Defective) | 0.7222 |
| F1 (Defective) | 0.7324 |
| Macro F1 | 0.7361 |
| OK recall | 0.75 |
| OK recall 95% CI | [0.4677, 0.9111] |
| Defective recall | 0.7222 |
| Defective recall 95% CI | [0.4677, 0.9111] |
| True negatives (OK→OK) | 27 |
| False positives (OK→Defective) | 9 |
| False negatives (Defective→OK) | 10 |
| True positives (Defective→Defective) | 26 |
| Unparseable responses | 0 |
| Distinct images | 12 OK / 12 Defective |
| Observations (image×prompt) | 72 |

> **Small-sample warning.** OK class has only 12 distinct images; one flip changes OK recall by 8.3 points. Treat OK recall as indicative, not statistically significant.
>
> Wilson 95% intervals use 12 distinct OK / 12 distinct Defective images, not the 36/36 (image, prompt) rows: repeated prompts on the same image are not independent samples.

### 4.2 Full held-out test set (178 images)

| Metric | Value |
|---|---|
| Accuracy | 0.633 |
| Balanced accuracy | 0.6873 |
| Precision (Defective) | 0.9719 |
| Recall (Defective) | 0.6245 |
| F1 (Defective) | 0.7604 |
| Macro F1 | 0.4882 |
| OK recall | 0.75 |
| OK recall 95% CI | [0.4677, 0.9111] |
| Defective recall | 0.6245 |
| Defective recall 95% CI | [0.5508, 0.6965] |
| True negatives (OK→OK) | 27 |
| False positives (OK→Defective) | 9 |
| False negatives (Defective→OK) | 187 |
| True positives (Defective→Defective) | 311 |
| Unparseable responses | 0 |
| Distinct images | 12 OK / 166 Defective |
| Observations (image×prompt) | 534 |

> The full set is 12 OK vs 166 Defective, so accuracy here is inflated by the majority class. Balanced accuracy and macro F1 are the honest reads.

### 4.3 Per-prompt (balanced slice)

| Prompt | Accuracy | Balanced acc | OK recall | Defective recall | Macro F1 | Unparseable |
|---|---|---|---|---|---|---|
| `prompt_1` | 0.7917 | 0.7917 | 0.6667 | 0.9167 | 0.7884 | 0 |
| `prompt_2` | 0.7083 | 0.7083 | 0.8333 | 0.5833 | 0.7037 | 0 |
| `prompt_3` | 0.7083 | 0.7084 | 0.75 | 0.6667 | 0.7078 | 0 |

### 4.4 Classification report (full test set)

```
              precision    recall  f1-score   support

          OK       0.13      0.75      0.22        36
   Defective       0.97      0.62      0.76       498

    accuracy                           0.63       534
   macro avg       0.55      0.69      0.49       534
weighted avg       0.91      0.63      0.72       534

```

Confusion matrices: `baseline_confusion_matrix.png` (full set), `baseline_confusion_matrix_balanced.png` (balanced slice).


## 5. TRACK B — SECONDARY: 12-class defect type

**Coverage: 0.4607** (246/534 responses yielded a recognisable class; 288 unmatched).

| Metric | Value |
|---|---|
| Accuracy | 0.122 |
| Macro precision | 0.0219 |
| Macro recall | 0.1304 |
| Macro F1 | 0.0371 |
| Weighted precision | 0.0171 |
| Weighted recall | 0.122 |
| Weighted F1 | 0.0299 |

> Metrics below are computed only on rows where the model produced a recognisable class. This subset is self-selected, so the scores are optimistically biased; read them together with coverage.

### 5.1 Per-class precision / recall / F1

```
                   precision    recall  f1-score   support

               ok       0.13      0.96      0.22        28
        cold_shut       0.00      0.00      0.00        26
            crack       0.00      0.00      0.00        14
             dent       0.00      0.00      0.00        29
            flash       0.00      0.00      0.00        39
        inclusion       0.14      0.60      0.22         5
    mixed_defects       0.00      0.00      0.00        11
          pinhole       0.00      0.00      0.00        39
         porosity       0.00      0.00      0.00         7
          scratch       0.00      0.00      0.00        14
        shrinkage       0.00      0.00      0.00         2
surface_roughness       0.00      0.00      0.00        32

         accuracy                           0.12       246
        macro avg       0.02      0.13      0.04       246
     weighted avg       0.02      0.12      0.03       246

```

Confusion matrix: `baseline_defect_type_confusion_matrix.png`


## 6. Error analysis

- Total observations: **534**
- Correct: **338** | Incorrect: **196**

| Error type | Count |
|---|---|
| `correct` | 338 |
| `false_negative_missed_defect` | 187 |
| `false_positive_false_alarm` | 9 |

| Failure flag | Count |
|---|---|
| `confidently_wrong` | 95 |
| `hallucinated_defect_type` | 1 |
| `low_confidence` | 3 |

Full per-row detail: `results/baseline/error_analysis.csv`


## 7. Example outputs

### 7.1 Correct predictions

**`cold_shut/cold_shut_00004.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_1` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 85.0

> Classification: Defective
> Reason: A wavy, dashed line is superimposed on the casting, suggesting a non-physical defect or artifact, not a real feature of the part. The casting appears to have internal voids or porosity visible through the center, which is a …

**`cold_shut/cold_shut_00009.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_1` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 90.0

> Classification: Defective
> Reason: A wavy, dashed line is superimposed across the casting, indicating a defect such as porosity, inclusions, or surface irregularity that deviates from the expected uniformity.
> Confidence: 90%

**`cold_shut/cold_shut_00009.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_2` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 30.0

> Defect present: Yes
> Defect type: Unknown
> Evidence: The image shows a circular casting with a central bore and concentric rings. A wavy dashed line traverses the image, possibly indicating a measurement or inspection path, but no clear, identifiable defect …

**`cold_shut/cold_shut_00009.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_3` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: n/a

> Prediction: Defective
> Evidence: The image shows a casting with a central bore and concentric rings. A dashed line traces a wavy path across the inner region, which appears to be a defect — possibly a void, porosity, or inclusion — that disrupts the uniformit…


### 7.2 Incorrect predictions

**`cold_shut/cold_shut_00004.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 100.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a metallic casting with concentric rings and a central bore, appearing structurally sound with no visible cracks, porosity, inclusions, or surface irregularities that would indicate a defect…

**`cold_shut/cold_shut_00004.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_3` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: n/a

> Prediction: OK
> Evidence: The image shows a metallic casting with concentric circular features, likely a bearing or similar mechanical component. There are no visible cracks, porosity, inclusions, or surface defects that would indicate a manufacturing defect.…

**`cold_shut/cold_shut_00027.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 100.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a metallic casting with a concentric circular structure, likely a bearing or similar component. There are no visible cracks, porosity, inclusions, or surface irregularities that would indica…

**`cold_shut/cold_shut_00027.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_3` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: n/a

> Prediction: OK
> Evidence: The image shows a casting with a concentric circular feature (likely a bore or internal cavity) and a smooth, uniform surface. There are no visible cracks, porosity, inclusions, or dimensional deviations that would indicate a defect.…


### 7.3 Missed defects — said OK for a Defective casting (false negatives)

_187 of 534 observations._

**`cold_shut/cold_shut_00004.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 100.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a metallic casting with concentric rings and a central bore, appearing structurally sound with no visible cracks, porosity, inclusions, or surface irregularities that would indicate a defect…

**`cold_shut/cold_shut_00004.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_3` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: n/a

> Prediction: OK
> Evidence: The image shows a metallic casting with concentric circular features, likely a bearing or similar mechanical component. There are no visible cracks, porosity, inclusions, or surface defects that would indicate a manufacturing defect.…

**`cold_shut/cold_shut_00027.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 100.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a metallic casting with a concentric circular structure, likely a bearing or similar component. There are no visible cracks, porosity, inclusions, or surface irregularities that would indica…

**`cold_shut/cold_shut_00027.jpg`** — truth: **Defective** (type: `cold_shut`) | prompt: `prompt_3` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: n/a

> Prediction: OK
> Evidence: The image shows a casting with a concentric circular feature (likely a bore or internal cavity) and a smooth, uniform surface. There are no visible cracks, porosity, inclusions, or dimensional deviations that would indicate a defect.…


### 7.4 False alarms — said Defective for an OK casting (false positives)

_9 of 534 observations._

**`ok/ok_00000.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_1` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 85.0

> Classification: Defective
> Reason: Visible internal porosity or voids in the center of the casting, indicating incomplete filling or gas inclusion during the casting process.
> Confidence: 85%

**`ok/ok_00005.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_1` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 95.0

> Classification: Defective
> Reason: A large, dark, irregularly shaped void or inclusion is visible in the center of the casting, indicating a manufacturing defect.
> Confidence: 95%

**`ok/ok_00005.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_2` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: 60.0

> Defect present: Yes
> Defect type: Unknown
> Evidence: The image shows a circular metallic component with a central hole and concentric rings, possibly a bearing or shaft. There is a noticeable dark, irregularly shaped area near the center, which may indicate …

**`ok/ok_00005.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_3` | parsed: **Defective** | type predicted: `Unknown` -> `Unmatched` | confidence: n/a

> Prediction: Defective
> Evidence: The image shows a casting with a dark, irregularly shaped void or inclusion in the center, which is not consistent with a properly solidified casting. The presence of this void suggests a manufacturing defect such as porosity …


### 7.5 Hallucinated defect types on OK castings

_A defect type asserted on a casting that is genuinely OK._

_28 of 534 observations._

**`ok/ok_00000.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 90.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a circular industrial casting with concentric rings and a central bore. The surface appears smooth and uniform, with no visible cracks, porosity, inclusions, or surface irregularities that w…

**`ok/ok_00000.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_3` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: n/a

> Prediction: OK
> Evidence: The image shows a circular casting with concentric rings and a central bore. There are no visible cracks, porosity, inclusions, or surface defects that would indicate a manufacturing defect. The surfaces appear smooth and uniform, co…

**`ok/ok_00024.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_1` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 95.0

> Classification: OK
> Reason: The casting shows no visible cracks, porosity, or surface defects; the concentric rings and internal structure appear uniform and well-defined, consistent with a properly manufactured part.
> Confidence: 95%

**`ok/ok_00024.jpg`** — truth: **OK** (type: `ok`) | prompt: `prompt_2` | parsed: **OK** | type predicted: `nan` -> `ok` | confidence: 90.0

> Defect present: No
> Defect type: Unknown
> Evidence: The image shows a metallic casting with a smooth, concentric circular structure and no visible cracks, porosity, inclusions, or surface irregularities that would indicate a defect. The internal bearing-like…


### 7.6 Unparseable responses

_0 of 534 observations. These are counted as errors, not discarded._

_None found._


## 8. Observations

- The base model produced a parseable OK/Defective verdict in **100.0%** of generations.
- Format compliance (an explicit labelled field): **100.0%**.
- On the balanced slice it called **12 OK** and **12 Defective** images; OK recall 0.75 vs Defective recall 0.7222.
- Track B coverage was **0.4607** — the base model rarely names a defect type matching the dataset's vocabulary, which is expected since the prompts never showed it that vocabulary.

_Interpretation to be written up in `reports/final_report.md`._


## 9. Limitations

- **Confidence is model-reported, NOT a calibrated probability.** No calibration was performed; treat every confidence figure as the model's own claim.
- **Small OK sample.** 12 distinct OK images in the balanced slice; one flip moves OK recall by ~8 points. OK recall is indicative, not statistically significant.
- **Synthetic defect labels.** The 12 defect classes were generated programmatically by painting defects onto real OK castings. Track B measures recognition of synthetic defect textures, NOT real industrial defect recognition.
- **Repeated prompts are not independent samples**; confidence intervals use distinct-image counts.
- This is a **research prototype**, not a certified industrial QC system.
