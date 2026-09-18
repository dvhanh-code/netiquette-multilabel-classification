"""
src/llm/output_parser.py
------------------------
Robust JSON parsing for LLM multilabel classification outputs.

Parsing hierarchy (stops at first success):
  1. json.loads on the full response string
  2. Extract the first {...} block via regex, then json.loads
  3. Field-by-field regex extraction as a last resort

Any failure returns a ParseResult with success=False and raw_text preserved,
so callers can log and skip without crashing.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

LABELS = ["hate_speech", "toxic", "threat", "insult"]

_REQUIRED_KEYS = set(LABELS) | {f"{l}_confidence" for l in LABELS}
_INT_KEYS = set(LABELS)
_FLOAT_KEYS = {f"{l}_confidence" for l in LABELS}


@dataclass
class ParseResult:
    success: bool
    predictions: Dict[str, int] = field(default_factory=dict)   # label → 0|1
    confidences: Dict[str, float] = field(default_factory=dict) # label → [0,1]
    reasoning: str = ""
    raw_text: str = ""
    error: str = ""


def parse_response(raw_text: str) -> ParseResult:
    """
    Parse a raw LLM response string into a ParseResult.

    Tries three strategies in order:
      1. Direct json.loads
      2. Regex extraction of first {...} block
      3. Field-by-field regex (partial recovery)

    Always returns a ParseResult; never raises.
    """
    text = raw_text.strip()

    # Strategy 1: direct parse
    parsed = _try_json_loads(text)
    if parsed is None:
        # Strategy 2: extract first JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            parsed = _try_json_loads(match.group(0))

    if parsed is not None:
        return _build_result(parsed, raw_text)

    # Strategy 3: field-by-field extraction
    result = _regex_extraction(text)
    if result.success:
        return result

    return ParseResult(
        success=False,
        raw_text=raw_text,
        error="All parsing strategies failed",
    )


def _try_json_loads(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _build_result(obj: dict, raw_text: str) -> ParseResult:
    """Validate and extract fields from a successfully parsed dict."""
    predictions: Dict[str, int] = {}
    confidences: Dict[str, float] = {}
    errors: List[str] = []

    for label in LABELS:
        # Binary prediction
        pred_val = obj.get(label)
        if pred_val is None:
            errors.append(f"missing key: {label}")
            predictions[label] = 0  # safe default
        else:
            try:
                predictions[label] = int(bool(pred_val))
            except (ValueError, TypeError):
                errors.append(f"invalid value for {label}: {pred_val!r}")
                predictions[label] = 0

        # Confidence
        conf_key = f"{label}_confidence"
        conf_val = obj.get(conf_key)
        if conf_val is None:
            # Infer confidence from binary prediction if missing
            confidences[label] = float(predictions[label])
        else:
            try:
                conf = float(conf_val)
                confidences[label] = float(np.clip(conf, 0.0, 1.0))
            except (ValueError, TypeError):
                errors.append(f"invalid confidence for {label}: {conf_val!r}")
                confidences[label] = float(predictions[label])

    reasoning = str(obj.get("reasoning", ""))

    return ParseResult(
        success=len(errors) == 0,
        predictions=predictions,
        confidences=confidences,
        reasoning=reasoning,
        raw_text=raw_text,
        error="; ".join(errors) if errors else "",
    )


def _regex_extraction(text: str) -> ParseResult:
    """Last-resort field-by-field extraction."""
    predictions: Dict[str, int] = {}
    confidences: Dict[str, float] = {}
    found_any = False

    for label in LABELS:
        # Match: "hate_speech": 1 or "hate_speech":1
        pred_match = re.search(
            rf'"{re.escape(label)}"\s*:\s*([01])',
            text,
        )
        if pred_match:
            predictions[label] = int(pred_match.group(1))
            found_any = True
        else:
            predictions[label] = 0

        conf_match = re.search(
            rf'"{re.escape(label)}_confidence"\s*:\s*([0-9]*\.?[0-9]+)',
            text,
        )
        if conf_match:
            confidences[label] = float(np.clip(float(conf_match.group(1)), 0.0, 1.0))
        else:
            confidences[label] = float(predictions[label])

    if not found_any:
        return ParseResult(success=False, raw_text=text, error="regex extraction found no fields")

    return ParseResult(
        success=True,
        predictions=predictions,
        confidences=confidences,
        reasoning="",
        raw_text=text,
        error="partial parse via regex",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch conversion: list of ParseResult → numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def results_to_arrays(
    results: List[ParseResult],
    label_names: List[str] = LABELS,
) -> Dict[str, np.ndarray]:
    """
    Convert a list of ParseResult objects to numpy arrays compatible with
    the existing evaluation pipeline (bootstrap_ci, transformer_metrics).

    Failed parses become NaN rows — they are masked out in evaluation,
    consistent with the dataset's NaN-annotation handling.

    Returns:
        {
          "predictions":  shape (n, len(labels)), float — 0.0 / 1.0 / NaN
          "confidences":  shape (n, len(labels)), float — [0,1] / NaN
          "success_mask": shape (n,), bool
        }
    """
    n = len(results)
    k = len(label_names)

    preds = np.full((n, k), fill_value=np.nan)
    confs = np.full((n, k), fill_value=np.nan)
    success_mask = np.zeros(n, dtype=bool)

    for i, r in enumerate(results):
        if not r.success and not r.predictions:
            continue
        success_mask[i] = r.success
        for j, label in enumerate(label_names):
            if label in r.predictions:
                preds[i, j] = float(r.predictions[label])
            if label in r.confidences:
                confs[i, j] = r.confidences[label]

    return {
        "predictions": preds,
        "confidences": confs,
        "success_mask": success_mask,
    }