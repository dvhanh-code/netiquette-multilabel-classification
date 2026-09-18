"""
src/llm/prompt_registry.py
--------------------------
Model-independent prompt registry for Ollama-style single-string inference.

This module is separate from src/llm/prompts.py, which is the Gemini-focused
system/user prompt builder with confidence scores and platform rules.
This registry serves the Ollama pipeline using simpler flat-string prompts.

Prompt variants
---------------
joint_basic      Minimal prompt: format only, no definitions.
joint_definitions Explicit German definitions per label (as specified by task design).
joint_fewshot    Definitions + 5 curated examples covering all multi-label cases.
joint_rules      Definitions + explicit independence rules (anti label-suppression).
joint_selfcheck  Two-step CoT: evaluate each label, verify coverage, return JSON.

Per-label prompts (for per_label mode)
---------------------------------------
One binary yes/no question per label; avoids the joint label-suppression bias
documented in Ma et al. (2025, EMNLP).

Prompt adapters
---------------
Adapters are lightweight string-level transformations applied after filling the
{text} slot. Each adapter key matches the prompt_adapter field in ModelConfig.

  default      — no modification
  qwen3        — append /no_think to disable Qwen3 chain-of-thought mode
  llama        — prepend a conciseness directive (reduces safety over-refusals)
  gemma        — append a strict JSON-only instruction (suppresses prose additions)
  deepseek_r1  — no modification (reasoning stripped at parse time, not prompt time)
"""

from typing import Callable, Dict, FrozenSet

LABELS = ["hate_speech", "toxic", "threat", "insult"]

# ─────────────────────────────────────────────────────────────────────────────
# Joint prompts
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS: Dict[str, str] = {
    "joint_basic": (
        "Du klassifizierst deutsche Online-Kommentare fuer ein Forschungsprojekt.\n"
        "Antworte NUR mit JSON, keine Erklaerung.\n"
        "\n"
        'Format: {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "Werte: 0=nein, 1=ja\n"
        "\n"
        'Kommentar: "{text}"'
    ),

    "joint_definitions": (
        "Du klassifizierst deutsche Online-Kommentare fuer ein Forschungsprojekt.\n"
        "Antworte NUR mit JSON, keine Erklaerung.\n"
        "\n"
        'Format: {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "\n"
        "Definitionen:\n"
        "- hate_speech: Angriff auf Gruppen aufgrund Herkunft, Religion, Geschlecht,"
        " Nationalitaet oder aehnlicher Merkmale.\n"
        "- toxic: allgemein feindselige, abwertende oder aggressive Sprache.\n"
        "- threat: explizite Androhung von Gewalt, Schaden oder Konsequenzen.\n"
        "- insult: direkte persoenliche Beleidigung oder Herabwuerdigung einer Person.\n"
        "\n"
        'Kommentar: "{text}"'
    ),

    "joint_fewshot": (
        "Du klassifizierst deutsche Online-Kommentare fuer ein Forschungsprojekt.\n"
        "Antworte NUR mit JSON, keine Erklaerung.\n"
        "\n"
        'Format: {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "\n"
        "Definitionen:\n"
        "- hate_speech: Angriff auf Gruppen aufgrund Herkunft, Religion, Geschlecht,"
        " Nationalitaet oder aehnlicher Merkmale.\n"
        "- toxic: allgemein feindselige, abwertende oder aggressive Sprache.\n"
        "- threat: explizite Androhung von Gewalt, Schaden oder Konsequenzen.\n"
        "- insult: direkte persoenliche Beleidigung oder Herabwuerdigung einer Person.\n"
        "\n"
        "Beispiele:\n"
        '1. "Diese Politik ist zum Kotzen, absolut widerlich." ->'
        ' {"hate_speech":0,"toxic":1,"threat":0,"insult":0}\n'
        "   (toxic aber kein insult: allgemeine Feindseligkeit, keine persoenliche Beleidigung)\n"
        '2. "Du bist ein kompletter Idiot und Versager." ->'
        ' {"hate_speech":0,"toxic":1,"threat":0,"insult":1}\n'
        "   (insult aber kein hate_speech: persoenliche Beleidigung ohne Gruppenmerkmale)\n"
        '3. "Alle Auslaender sollen verschwinden, die verseuchen unser Land." ->'
        ' {"hate_speech":1,"toxic":1,"threat":0,"insult":0}\n'
        "   (hate_speech UND toxic: Gruppenangriff mit feindseliger Sprache)\n"
        '4. "Ich werde dich finden und dir das beibringen, was du verdienst." ->'
        ' {"hate_speech":0,"toxic":1,"threat":1,"insult":0}\n'
        "   (explizite Bedrohung: Gewaltandrohung gegen eine Person)\n"
        '5. "Interessanter Beitrag, ich sehe das etwas anders aber danke fuers Teilen." ->'
        ' {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "   (harmloser Kommentar: keine negativen Kategorien)\n"
        "\n"
        'Kommentar: "{text}"'
    ),

    "joint_rules": (
        "Du klassifizierst deutsche Online-Kommentare fuer ein Forschungsprojekt.\n"
        "Antworte NUR mit JSON, keine Erklaerung.\n"
        "\n"
        'Format: {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "\n"
        "Definitionen:\n"
        "- hate_speech: Angriff auf Gruppen aufgrund Herkunft, Religion, Geschlecht,"
        " Nationalitaet oder aehnlicher Merkmale.\n"
        "- toxic: allgemein feindselige, abwertende oder aggressive Sprache.\n"
        "- threat: explizite Androhung von Gewalt, Schaden oder Konsequenzen.\n"
        "- insult: direkte persoenliche Beleidigung oder Herabwuerdigung einer Person.\n"
        "\n"
        "REGELN:\n"
        "1. Jedes Label ist eine UNABHAENGIGE binaere Entscheidung (0 oder 1).\n"
        "2. Ein Kommentar kann 0, 1, 2, 3 oder alle 4 Labels gleichzeitig erhalten.\n"
        "3. Unterdruecke kein Label aufgrund anderer Labels.\n"
        "4. hate_speech impliziert NICHT automatisch toxic (und umgekehrt).\n"
        "5. insult bezieht sich auf Personen, hate_speech auf Gruppen — beide koennen"
        " gleichzeitig zutreffen.\n"
        "6. Bewerte ausschliesslich den tatsaechlichen Inhalt, nicht die Absicht.\n"
        "\n"
        'Kommentar: "{text}"'
    ),

    "joint_selfcheck": (
        "Du klassifizierst deutsche Online-Kommentare fuer ein Forschungsprojekt.\n"
        "\n"
        "Definitionen:\n"
        "- hate_speech: Angriff auf Gruppen aufgrund Herkunft, Religion, Geschlecht,"
        " Nationalitaet oder aehnlicher Merkmale.\n"
        "- toxic: allgemein feindselige, abwertende oder aggressive Sprache.\n"
        "- threat: explizite Androhung von Gewalt, Schaden oder Konsequenzen.\n"
        "- insult: direkte persoenliche Beleidigung oder Herabwuerdigung einer Person.\n"
        "\n"
        "Vorgehensweise:\n"
        "Schritt 1: Bewerte jedes Label unabhaengig voneinander.\n"
        "Schritt 2: Pruefe ob alle 4 Labels im JSON vorhanden sind.\n"
        "Schritt 3: Gib am Ende NUR das JSON aus, kein weiterer Text danach.\n"
        "\n"
        'JSON-Format: {"hate_speech":0,"toxic":0,"threat":0,"insult":0}\n'
        "\n"
        'Kommentar: "{text}"'
    ),
}

PROMPT_VARIANTS: FrozenSet[str] = frozenset(PROMPTS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Per-label prompts (Ma et al. 2025 — avoids label-suppression bias)
# ─────────────────────────────────────────────────────────────────────────────

PER_LABEL_PROMPTS: Dict[str, str] = {
    "hate_speech": (
        "Enthaelt dieser Kommentar Angriffe auf Gruppen aufgrund von"
        " Herkunft/Religion/Geschlecht/Nationalitaet? Antworte NUR mit 0 oder 1.\n\n"
        'Kommentar: "{text}"\n\nAntwort:'
    ),
    "toxic": (
        "Enthaelt dieser Kommentar feindselige, abwertende oder aggressive Sprache?"
        " Antworte NUR mit 0 oder 1.\n\n"
        'Kommentar: "{text}"\n\nAntwort:'
    ),
    "threat": (
        "Enthaelt dieser Kommentar eine explizite Androhung von Gewalt, Schaden"
        " oder Konsequenzen? Antworte NUR mit 0 oder 1.\n\n"
        'Kommentar: "{text}"\n\nAntwort:'
    ),
    "insult": (
        "Enthaelt dieser Kommentar eine direkte persoenliche Beleidigung"
        " oder Herabwuerdigung einer Person? Antworte NUR mit 0 oder 1.\n\n"
        'Kommentar: "{text}"\n\nAntwort:'
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt adapters
# ─────────────────────────────────────────────────────────────────────────────
# Each adapter is a pure function str → str applied after {text} is filled.
# Rationale for each is in the module docstring above.

def _adapt_qwen3(prompt: str) -> str:
    """Append /no_think to disable Qwen3 thinking mode (native Qwen3 feature)."""
    return prompt + "\n/no_think"


def _adapt_llama(prompt: str) -> str:
    """
    Prepend a conciseness directive.

    Llama 3.x safety tuning sometimes causes the model to refuse classifying
    explicit content and return a prose refusal instead of JSON.  A direct,
    concise preamble reduces this without compromising classification quality.
    """
    return "Sei praezise. Antworte NUR mit gueltigem JSON.\n\n" + prompt


def _adapt_gemma(prompt: str) -> str:
    """
    Append a strict JSON-only instruction.

    Gemma 3 tends to add prose explanations after the JSON block.
    The suffix reinforces the output constraint and is picked up by the parser's
    'last JSON block' strategy even if Gemma still adds trailing text.
    """
    return prompt + "\nAntworte ausschliesslich mit gueltigem JSON, kein weiterer Text."


def _adapt_deepseek_r1(prompt: str) -> str:
    """
    No-op: DeepSeek-R1 always reasons via <think>...</think>.

    The adaptation happens in the parser (RobustJSONParser._strip_reasoning_blocks),
    not in the prompt, so the model's native reasoning flow is preserved.
    """
    return prompt


ADAPTERS: Dict[str, Callable[[str], str]] = {
    "default":      lambda p: p,
    "qwen3":        _adapt_qwen3,
    "llama":        _adapt_llama,
    "gemma":        _adapt_gemma,
    "deepseek_r1":  _adapt_deepseek_r1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    text: str,
    variant: str = "joint_basic",
    adapter: str = "default",
) -> str:
    """
    Build a ready-to-send prompt string.

    Args:
        text:     The German comment to classify.
        variant:  Key into PROMPTS (e.g. 'joint_fewshot').
        adapter:  Key into ADAPTERS (from ModelConfig.prompt_adapter).

    Returns:
        The fully constructed prompt string.
    """
    template = PROMPTS.get(variant, PROMPTS["joint_basic"])
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    prompt = template.replace("{text}", safe_text)
    adapt_fn = ADAPTERS.get(adapter, ADAPTERS["default"])
    return adapt_fn(prompt)


def build_per_label_prompt(text: str, label: str) -> str:
    """Build the per-label binary prompt for a single label."""
    if label not in PER_LABEL_PROMPTS:
        raise ValueError(f"Unknown label '{label}'. Valid: {list(PER_LABEL_PROMPTS)}")
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    return PER_LABEL_PROMPTS[label].replace("{text}", safe_text)