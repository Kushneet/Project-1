"""Centralized prompts and response parsing.

FROZEN CONTRACT
---------------
The three evaluation prompts below are used *verbatim* for both the base-model
baseline (Phase 4) and the fine-tuned evaluation (Phase 10). Changing them
after the baseline has been run invalidates the comparison. If a prompt must
change, re-run the baseline.

Note on defect types: the prompts deliberately do NOT enumerate the dataset's
class names. The base model has never seen them, so injecting the label set
would hand it the answer space and inflate the baseline unfairly.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------
# Evaluation prompts — DO NOT EDIT after the baseline run.
# --------------------------------------------------------------------------

PROMPT_1 = """You are an industrial casting quality inspection assistant.

Inspect the uploaded casting image carefully.

Determine whether the casting is:

1. OK
2. Defective

Return your answer in this format:

Classification: OK or Defective
Reason: brief visual explanation
Confidence: 0-100%"""

PROMPT_2 = """Inspect this industrial casting image for visible defects.

Determine whether a defect is present.

If the available dataset labels support a specific defect type, identify it.
If a specific defect type cannot be established, write Unknown.

Return:

Defect present: Yes or No
Defect type: [label or Unknown]
Evidence: brief explanation
Confidence: 0-100%"""

PROMPT_3 = """You are performing visual quality inspection of a manufactured casting.

Carefully inspect the image and classify it as OK or Defective.

Do not invent information that cannot be visually supported.

Return:

Prediction:
Evidence:
Confidence:"""

EVAL_PROMPTS: dict[str, str] = {
    "prompt_1": PROMPT_1,
    "prompt_2": PROMPT_2,
    "prompt_3": PROMPT_3,
}

# --------------------------------------------------------------------------
# Instruction templates used to build the fine-tuning set (Phase 6).
# Varied phrasing prevents the model from overfitting one question string.
# --------------------------------------------------------------------------

TRAIN_INSTRUCTIONS: list[str] = [
    "Is this casting defective or OK?",
    "Inspect this casting for visible defects.",
    "Perform quality inspection on this casting.",
    "Classify this casting image.",
    "Does this manufactured casting contain a defect?",
]

BINARY_OK = "OK"
BINARY_DEFECTIVE = "Defective"
UNKNOWN = "Unknown"
UNPARSEABLE = "Unparseable"


def get_prompt(prompt_id: str) -> str:
    """Return an evaluation prompt by id, failing loudly on a typo."""
    if prompt_id not in EVAL_PROMPTS:
        raise KeyError(f"Unknown prompt_id {prompt_id!r}; valid: {sorted(EVAL_PROMPTS)}")
    return EVAL_PROMPTS[prompt_id]


def build_messages(prompt_id: str, image: Any = None) -> list[dict[str, Any]]:
    """Build a chat-template message list pairing one image with one prompt."""
    content: list[dict[str, Any]] = [{"type": "image"}]
    if image is not None:
        content = [{"type": "image", "image": image}]
    content.append({"type": "text", "text": get_prompt(prompt_id)})
    return [{"role": "user", "content": content}]


def build_training_answer(label: str, is_defective: bool, defect_type: str | None = None) -> str:
    """Compose the ground-truth target string for one training example.

    The answer is derived *only* from the dataset label — never invented.
    """
    classification = BINARY_DEFECTIVE if is_defective else BINARY_OK
    lines = [f"Classification: {classification}"]
    if is_defective and defect_type and defect_type.lower() not in {"unknown", ""}:
        lines.append(f"Defect type: {defect_type}")
    else:
        lines.append(f"Defect type: {UNKNOWN if is_defective else 'None'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_DEFECTIVE_PAT = re.compile(
    r"\b(defect(?:ive|s|ed)?|not\s+ok|faulty|damaged|reject(?:ed)?|ng)\b", re.I
)
_OK_PAT = re.compile(r"\b(ok|okay|no\s+defect|defect\s*free|acceptable|good|pass(?:ed)?|normal)\b", re.I)


def _find_field(text: str, *names: str) -> str | None:
    """Return the value following a 'Field: value' line, if present."""
    for name in names:
        m = re.search(rf"^\s*{name}\s*[:\-]\s*(.+)$", text, re.I | re.M)
        if m:
            value = m.group(1).strip()
            value = re.sub(r"^\**|\**$", "", value).strip()  # strip markdown bold
            if value:
                return value
    return None


def parse_confidence(text: str) -> float | None:
    """Extract a model-reported confidence in 0-100.

    NOTE: this is the model's own claim, not a calibrated probability.
    """
    value = _find_field(text, "confidence", "confidence level")
    candidate = value if value is not None else text
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", candidate)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*100", candidate)
    if not m and value is not None:
        m = re.search(r"(\d+(?:\.\d+)?)", value)
    if not m:
        return None
    conf = float(m.group(1))
    if 0.0 < conf <= 1.0 and "%" not in m.group(0):
        conf *= 100.0  # a 0-1 probability
    return max(0.0, min(100.0, conf))


def _classify_text(text: str) -> str:
    """Decide OK vs Defective from free text, preferring the earliest verdict."""
    d = _DEFECTIVE_PAT.search(text)
    o = _OK_PAT.search(text)
    if d and o:
        # "no defect" style phrasing makes _OK_PAT match inside a defect word;
        # the earlier match wins.
        return BINARY_DEFECTIVE if d.start() < o.start() else BINARY_OK
    if d:
        return BINARY_DEFECTIVE
    if o:
        return BINARY_OK
    return UNPARSEABLE


def parse_prediction(raw_response: str) -> dict[str, Any]:
    """Parse a raw model response into a structured prediction.

    Returns keys: prediction, defect_type, confidence, evidence, format_ok.
    ``prediction`` is "OK", "Defective", or "Unparseable" — never guessed.
    """
    if not raw_response or not raw_response.strip():
        return {
            "prediction": UNPARSEABLE,
            "defect_type": UNKNOWN,
            "confidence": None,
            "evidence": "",
            "format_ok": False,
        }

    text = raw_response.strip()

    # 1) Preferred path: an explicit labelled field.
    field = _find_field(text, "classification", "prediction", "defect present", "answer")
    format_ok = field is not None
    if field is not None:
        head = field.split("\n")[0]
        if re.match(r"^\s*(yes|no)\b", head, re.I):
            # "Defect present: Yes/No" inverts the polarity.
            prediction = BINARY_DEFECTIVE if re.match(r"^\s*yes", head, re.I) else BINARY_OK
        else:
            prediction = _classify_text(head)
        if prediction == UNPARSEABLE:
            prediction = _classify_text(text)
    else:
        prediction = _classify_text(text)

    raw_type = _find_field(text, "defect type", "defect_type", "type of defect")
    defect_type = (raw_type or "").split("\n")[0].strip().strip("[]").strip()
    # An absent, empty or placeholder defect type is resolved from the verdict:
    # an OK casting has no defect type, a defective one has an unidentified type.
    if not defect_type or defect_type.lower() in {"n/a", "na", "none", "-", "unknown"}:
        defect_type = "None" if prediction == BINARY_OK else UNKNOWN

    evidence = _find_field(text, "evidence", "reason", "explanation", "reasoning") or ""

    return {
        "prediction": prediction,
        "defect_type": defect_type,
        "confidence": parse_confidence(text),
        "evidence": evidence.split("\n")[0].strip(),
        "format_ok": bool(format_ok),
    }


# --------------------------------------------------------------------------
# TRACK B — defect-type canonicalisation
# --------------------------------------------------------------------------
#
# The evaluation prompts deliberately do not list the dataset's class names, so
# the BASE model emits free-form descriptions ("surface pitting", "gas holes").
# To score Track B we map that free text onto the 12 dataset labels. A response
# that matches nothing maps to UNMATCHED and is reported as such — never
# silently forced into a class.

DEFECT_CLASSES: list[str] = [
    "ok", "cold_shut", "crack", "dent", "flash", "inclusion", "mixed_defects",
    "pinhole", "porosity", "scratch", "shrinkage", "surface_roughness",
]

UNMATCHED = "Unmatched"

# Synonyms are matched longest-first so "cold shut" wins over "shut".
_DEFECT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cold_shut": ("cold shut", "cold-shut", "coldshut", "cold lap", "misrun",
                  "incomplete fusion", "unfused", "lack of fusion"),
    "crack": ("crack", "cracking", "fracture", "fissure", "split", "rupture", "tear"),
    "dent": ("dent", "dented", "indentation", "depression", "impact mark", "ding"),
    "flash": ("flash", "fin", "burr", "parting line", "excess material", "overflow"),
    "inclusion": ("inclusion", "foreign material", "foreign particle", "slag",
                  "dross", "embedded particle", "contaminant"),
    "mixed_defects": ("mixed defect", "mixed_defects", "multiple defect",
                      "several defect", "combination of defect", "various defect"),
    "pinhole": ("pinhole", "pin hole", "pin-hole", "gas hole", "small hole",
                "tiny hole", "needle hole"),
    "porosity": ("porosity", "porous", "pore", "void", "gas pocket", "blowhole",
                 "blow hole", "cavity", "bubble", "pitting", "pitted"),
    "scratch": ("scratch", "scratches", "scoring", "score mark", "abrasion",
                "linear mark", "streak", "scrape"),
    "shrinkage": ("shrinkage", "shrink", "sink mark", "solidification cavity",
                  "contraction"),
    "surface_roughness": ("surface roughness", "rough surface", "roughness",
                          "rough texture", "texture variation", "uneven surface",
                          "coarse surface"),
    "ok": ("no defect", "defect free", "defect-free", "none", "not applicable",
           "n/a", "ok", "acceptable", "no visible defect"),
}


def canonicalize_defect_type(text: str | None, prediction: str | None = None) -> str:
    """Map a free-text defect description onto one of the 12 dataset labels.

    Returns a class name, ``"ok"`` when the casting was judged OK, or
    ``UNMATCHED`` when nothing matches. Never guesses a class.
    """
    if prediction == BINARY_OK:
        return "ok"
    if not text:
        return UNMATCHED

    low = str(text).strip().lower()
    if not low or low in {"unknown", "none", "n/a", "na", "-", "unparseable"}:
        return UNMATCHED

    # Exact label match first (the fine-tuned model should hit this path).
    squashed = low.replace(" ", "_").replace("-", "_")
    if squashed in DEFECT_CLASSES:
        return squashed

    # Otherwise the longest matching synonym wins.
    best: tuple[int, str] | None = None
    for cls, synonyms in _DEFECT_SYNONYMS.items():
        for syn in synonyms:
            if syn in low and (best is None or len(syn) > best[0]):
                best = (len(syn), cls)
    return best[1] if best else UNMATCHED
