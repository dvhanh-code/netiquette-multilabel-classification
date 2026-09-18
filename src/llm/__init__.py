"""
src/llm
-------
LLM-based multilabel netiquette classification.

Gemini pipeline (existing — unchanged):
    prompts        — PromptBuilder with label defs, platform rules, CoT
    output_parser  — ParseResult, parse_response, results_to_arrays
    gemini_client  — GeminiClient with rate limiting, retry, JSONL cache
    inference      — LLMInferenceEngine: DataFrame → numpy arrays (Gemini)
    platform_rules — Platform presets (strict, lenient, news_comments, …)

Ollama / multi-model pipeline (new):
    backends       — LLMBackend ABC + OllamaBackend
    model_configs  — ModelConfig dataclass + MODEL_CONFIGS registry
    prompt_registry — PROMPTS dict, PromptAdapter functions, build_prompt()
    parsing        — RobustJSONParser (5-strategy, DeepSeek/Qwen3 aware)
    checkpoint     — Resume-safe JSONL checkpoint I/O
    engine         — OllamaInferenceEngine with latency tracking
"""

# ── Gemini pipeline (preserve existing imports) ───────────────────────────────
# Do not re-export here to avoid pulling in google-genai at import time.
# Callers use: from src.llm.inference import LLMInferenceEngine

# ── Ollama / multi-model pipeline ─────────────────────────────────────────────
from .backends import GenerationConfig, LLMBackend, OllamaBackend
from .model_configs import MODEL_CONFIGS, ModelConfig, get_model_config
from .prompt_registry import (
    LABELS,
    PER_LABEL_PROMPTS,
    PROMPT_VARIANTS,
    PROMPTS,
    build_per_label_prompt,
    build_prompt,
)
from .parsing import RobustJSONParser, _FALLBACK, parse_binary
from .checkpoint import append_checkpoint, load_checkpoint
from .engine import BatchStats, InferenceResult, OllamaInferenceEngine

__all__ = [
    # Backends
    "GenerationConfig",
    "LLMBackend",
    "OllamaBackend",
    # Config
    "ModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    # Prompts
    "LABELS",
    "PROMPTS",
    "PER_LABEL_PROMPTS",
    "PROMPT_VARIANTS",
    "build_prompt",
    "build_per_label_prompt",
    # Parsing
    "RobustJSONParser",
    "parse_binary",
    "_FALLBACK",
    # Checkpoint
    "load_checkpoint",
    "append_checkpoint",
    # Engine
    "OllamaInferenceEngine",
    "InferenceResult",
    "BatchStats",
]
