"""
src/llm/parsing.py
------------------
Robust JSON extraction for Ollama model outputs.

Parsing strategy (applied in order, stops at first success):

  1. direct          json.loads on the full (cleaned) response string
  2. markdown_fence  strip ```json … ``` fences, then json.loads
  3. last_json_block findall of all {…} blocks, try last → first
                     (self-check variants put JSON at the end of their output)
  4. first_json_block regex for the first {…} block (fallback)
  5. field_by_field   per-key regex extraction (last resort; partial recovery)

Value coercion in _validate():
  - int / float        → int(bool(val))      e.g. 2 → 1, 0.0 → 0
  - bool               → int(val)            True → 1, False → 0
  - "1"/"true"/"yes"   → 1
  - "0"/"false"/"no"   → 0
  - missing key        → 0  (with logger.warning, per thesis spec)

DeepSeek-R1 / Qwen3 reasoning blocks:
  <think>…</think> blocks are stripped before any parsing strategy runs.
  This ensures reasoning models work without any prompt-side changes.

This module is separate from src/llm/output_parser.py, which serves the
Gemini pipeline and expects confidence scores in a different schema.
"""

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

LABELS = ["hate_speech", "toxic", "threat", "insult"]
_FALLBACK: Dict[str, int] = {label: 0 for label in LABELS}


# ─────────────────────────────────────────────────────────────────────────────
# Public parser
# ─────────────────────────────────────────────────────────────────────────────

class RobustJSONParser:
    """
    Multi-strategy JSON extractor for Ollama LLM outputs.

    Usage:
        parser = RobustJSONParser()
        pred = parser.parse(raw_response)
        # pred is {label: 0|1} or None if all strategies fail
    """

    def parse(self, raw: str) -> Optional[Dict[str, int]]:
        """
        Extract a {label: 0|1} dict from a raw model response.

        Returns None only when every strategy fails — the caller should
        then retry or fall back to all-zeros.
        """
        if not raw:
            return None

        # Strip reasoning blocks before any strategy runs
        text = _strip_reasoning_blocks(raw.strip())

        for strategy in (
            self._parse_direct,
            self._parse_markdown_fence,
            self._parse_last_json_block,
            self._parse_first_json_block,
            self._parse_field_by_field,
        ):
            result = strategy(text)
            if result is not None:
                return result

        logger.debug("All parsing strategies failed. raw=%.100r", raw)
        return None

    # ── Strategies ─────────────────────────────────────────────────────────────

    def _parse_direct(self, text: str) -> Optional[Dict[str, int]]:
        obj = _try_json_loads(text)
        return _validate(obj) if obj is not None else None

    def _parse_markdown_fence(self, text: str) -> Optional[Dict[str, int]]:
        match = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
        if not match:
            return None
        obj = _try_json_loads(match.group(1).strip())
        return _validate(obj) if obj is not None else None

    def _parse_last_json_block(self, text: str) -> Optional[Dict[str, int]]:
        """Try all {…} blocks from last to first.

        Uses r'{[^{}]*}' which matches only non-nested braces, so Qwen3
        reasoning text ("Thinking...") is skipped because it contains no {}.
        Reasoning models and self-check variants place the final JSON at the
        end, so last→first order succeeds on the first valid attempt.
        """
        blocks = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
        for block in reversed(blocks):
            obj = _try_json_loads(block)
            if obj is not None:
                result = _validate(obj)
                if result is not None:
                    return result
        return None

    def _parse_first_json_block(self, text: str) -> Optional[Dict[str, int]]:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            return None
        obj = _try_json_loads(match.group(0))
        return _validate(obj) if obj is not None else None

    def _parse_field_by_field(self, text: str) -> Optional[Dict[str, int]]:
        """
        Last-resort field extraction via per-key regex.

        Recovers from responses like:
            hate_speech: 0, toxic: 1, threat: 0, insult: 1
        or partial JSON where braces are malformed.
        """
        result: Dict[str, int] = {}
        found_any = False

        for label in LABELS:
            match = re.search(
                rf'"{re.escape(label)}"\s*:\s*(true|false|[01])',
                text,
                re.IGNORECASE,
            )
            if match:
                val = match.group(1).lower()
                result[label] = 1 if val in ("1", "true") else 0
                found_any = True
            else:
                # Try unquoted form: hate_speech: 0
                match2 = re.search(
                    rf"\b{re.escape(label)}\b\s*:\s*([01])",
                    text,
                )
                if match2:
                    result[label] = int(match2.group(1))
                    found_any = True
                else:
                    result[label] = 0

        if not found_any:
            return None

        logger.debug("Used field-by-field extraction fallback")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-label binary parser (for per_label mode)
# ─────────────────────────────────────────────────────────────────────────────

def parse_binary(raw: str) -> int:
    """
    Extract 0 or 1 from a short per-label response.

    Tries a word-boundary match first (avoids false positives in '10', '01'),
    then falls back to scanning for any digit character.
    Defaults to 0 if the response is empty or contains neither 0 nor 1.
    """
    if not raw:
        return 0
    cleaned = _strip_reasoning_blocks(raw.strip())
    match = re.search(r"\b([01])\b", cleaned)
    if match:
        return int(match.group(1))
    for ch in cleaned:
        if ch in ("0", "1"):
            return int(ch)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_reasoning_blocks(text: str) -> str:
    """
    Remove reasoning preambles before parsing.

    Two formats handled:
      - <think>…</think>: DeepSeek-R1 and Qwen3 via direct API.
      - Thinking……done thinking.: Qwen3 via Ollama when /no_think is ignored.
    """
    # XML-style reasoning block (DeepSeek-R1, Qwen3 API)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Ollama Qwen3 plaintext reasoning block
    text = re.sub(r"Thinking\.\.\..*?\.\.\.done thinking\.", "", text, flags=re.DOTALL)
    return text.strip()


def _try_json_loads(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _validate(obj: dict) -> Optional[Dict[str, int]]:
    """
    Coerce a parsed JSON dict to {label: 0|1}.

    Rules:
    - bool   → int(val)
    - int/float → int(bool(val))
    - str "1"/"true"/"yes"/"ja" → 1
    - str "0"/"false"/"no"/"nein" → 0
    - missing label → 0 with logger.warning (does NOT return None)
    - no label keys at all → return None (invalid JSON, trigger retry)
    """
    if not isinstance(obj, dict):
        return None

    result: Dict[str, int] = {}
    found_any = False
    missing = []

    for label in LABELS:
        val = obj.get(label)

        if val is None:
            result[label] = 0
            missing.append(label)
            continue

        found_any = True

        if isinstance(val, bool):
            result[label] = int(val)
        elif isinstance(val, (int, float)):
            result[label] = int(bool(val))
        elif isinstance(val, str):
            v = val.lower().strip()
            if v in ("1", "true", "yes", "ja"):
                result[label] = 1
            elif v in ("0", "false", "no", "nein"):
                result[label] = 0
            else:
                try:
                    result[label] = int(bool(int(v)))
                except (ValueError, TypeError):
                    logger.warning(
                        "Cannot coerce value '%s' for label '%s'; defaulting to 0",
                        val, label,
                    )
                    result[label] = 0
        else:
            try:
                result[label] = int(bool(val))
            except (TypeError, ValueError):
                logger.warning(
                    "Cannot coerce value '%s' (type %s) for label '%s'; defaulting to 0",
                    val, type(val).__name__, label,
                )
                result[label] = 0

    if not found_any:
        # Object had no recognizable label keys — probably unrelated JSON
        return None

    if missing:
        logger.warning("Labels %s absent from JSON response; defaulting to 0", missing)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (run with: python3 src/llm/parsing.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Exact Qwen3 Ollama output format reported in production
    _RAW = (
        "Thinking...\n"
        "...\n"
        "...done thinking.\n"
        "\n"
        '{"hate_speech": 0, "toxic": 1, "threat": 0, "insult": 1}'
    )

    _parser = RobustJSONParser()
    _result = _parser.parse(_RAW)

    assert _result is not None, "Parser returned None on Qwen3 reasoning output"
    assert _result == {"hate_speech": 0, "toxic": 1, "threat": 0, "insult": 1}, (
        f"Unexpected result: {_result}"
    )
    print("Smoke test passed:", _result)
