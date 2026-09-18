"""
src/llm/model_configs.py
------------------------
Per-model generation parameters and prompt-adaptation strategy.

Design rationale
----------------
Different open-weight models have different optimal settings:

  qwen2.5:7b   — reliable JSON output at temperature=0; no special handling needed.
  qwen3:8b     — defaults to chain-of-thought "thinking" mode; /no_think suffix
                 disables it to force direct JSON output (Qwen3 native feature).
  llama3.1:8b  — safety-tuned; over-refusals on explicit content; shorter, more
                 direct prompts reduce refusals without sacrificing accuracy.
  gemma3:12b   — tends to add prose explanations; strict formatting instruction
                 appended to prompt reduces this.
  mistral-small — well-aligned to instruction format; default settings work well.
  deepseek-r1  — always reasons before answering (built-in CoT); parser strips
                 the <think>...</think> block; needs more tokens and longer timeout.

prompt_adapter keys map to functions in prompt_registry.ADAPTERS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """
    Complete configuration for one LLM model.

    Fields forwarded to GenerationConfig at engine build time:
        temperature, top_p, num_predict, timeout, stop

    Fields used by the prompt layer:
        prompt_adapter   — key into prompt_registry.ADAPTERS
        reasoning_mode   — when True, the parser strips <think>…</think> blocks

    Fields used by the inference loop:
        max_attempts     — retry limit before falling back to all-zeros
        description      — human-readable note (not used at runtime)
    """
    temperature: float = 0.0
    top_p: float = 1.0
    num_predict: int = 128
    timeout: int = 120
    stop: List[str] = field(default_factory=list)
    reasoning_mode: bool = False
    prompt_adapter: str = "default"
    max_attempts: int = 3
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen2.5:7b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=120,
        reasoning_mode=False,
        prompt_adapter="default",
        description="Qwen 2.5 7B — baseline; reliable JSON at temperature=0.",
    ),
    "qwen2.5:14b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=180,
        reasoning_mode=False,
        prompt_adapter="default",
        description="Qwen 2.5 14B — larger variant; same adapter as 7B.",
    ),
    "qwen3:8b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=768,
        timeout=150,
        reasoning_mode=True,
        prompt_adapter="qwen3",
        description=(
            "Qwen3 8B — /no_think suffix sent but not always honoured via Ollama; "
            "reasoning_mode=True strips 'Thinking...' blocks and <think> blocks. "
            "num_predict=768 accommodates reasoning preamble + JSON."
        ),
    ),
    "qwen3:14b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=768,
        timeout=210,
        reasoning_mode=True,
        prompt_adapter="qwen3",
        description="Qwen3 14B — larger variant; same reasoning handling as 8B.",
    ),
    "llama3.1:8b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=120,
        reasoning_mode=False,
        prompt_adapter="llama",
        description=(
            "Llama 3.1 8B — safety-tuned; shorter, direct prompts reduce "
            "over-refusals on hate-speech content."
        ),
    ),
    "llama3.2:3b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=90,
        reasoning_mode=False,
        prompt_adapter="llama",
        description="Llama 3.2 3B — fast baseline; same adapter as 3.1:8b.",
    ),
    "gemma3:12b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=180,
        reasoning_mode=False,
        prompt_adapter="gemma",
        description=(
            "Gemma 3 12B — tends to add prose; strict formatting suffix suppresses it."
        ),
    ),
    "gemma3:4b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=120,
        reasoning_mode=False,
        prompt_adapter="gemma",
        description="Gemma 3 4B — smaller variant; same adapter.",
    ),
    "mistral-small": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=120,
        reasoning_mode=False,
        prompt_adapter="default",
        description="Mistral Small — strong instruction following; default adapter.",
    ),
    "mistral:7b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=120,
        reasoning_mode=False,
        prompt_adapter="default",
        description="Mistral 7B v0.3 — default adapter.",
    ),
    "deepseek-r1": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=768,
        timeout=300,
        reasoning_mode=True,
        prompt_adapter="deepseek_r1",
        description=(
            "DeepSeek-R1 — mandatory chain-of-thought via <think>…</think>; "
            "reasoning_mode=True tells the parser to strip the thinking block. "
            "Needs more tokens and a longer timeout."
        ),
    ),
    "deepseek-r1:7b": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=512,
        timeout=240,
        reasoning_mode=True,
        prompt_adapter="deepseek_r1",
        description="DeepSeek-R1 distilled 7B — same reasoning adapter.",
    ),
    "phi4": ModelConfig(
        temperature=0.0,
        top_p=1.0,
        num_predict=128,
        timeout=150,
        reasoning_mode=False,
        prompt_adapter="default",
        description="Microsoft Phi-4 — default adapter.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_model_config(model: str) -> ModelConfig:
    """
    Return the ModelConfig for a given model tag.

    Lookup order:
    1. Exact match in MODEL_CONFIGS.
    2. Prefix match on the model family (e.g., 'qwen3:72b' → qwen3 adapter).
    3. Default ModelConfig() with a warning.
    """
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]

    # Prefix match: "qwen3:72b" → match "qwen3:8b" family adapter
    family_prefixes = {
        "qwen3": "qwen3",
        "qwen2.5": "default",
        "llama": "llama",
        "gemma": "gemma",
        "mistral": "default",
        "deepseek-r1": "deepseek_r1",
        "deepseek": "default",
        "phi": "default",
    }
    model_lower = model.lower()
    for prefix, adapter in family_prefixes.items():
        if model_lower.startswith(prefix):
            import logging
            logging.getLogger(__name__).info(
                "No exact config for '%s'; using family adapter '%s'", model, adapter
            )
            return ModelConfig(prompt_adapter=adapter, description=f"Auto-config for {model}")

    import logging
    logging.getLogger(__name__).warning(
        "No config found for '%s'; falling back to defaults. "
        "Add an entry to MODEL_CONFIGS for accurate settings.",
        model,
    )
    return ModelConfig()


def list_configured_models() -> List[str]:
    """Return all model tags with explicit configs."""
    return sorted(MODEL_CONFIGS.keys())