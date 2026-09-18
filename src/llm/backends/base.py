"""
src/llm/backends/base.py
------------------------
Abstract base class for LLM inference backends.

Adding a new provider (OpenAI, Claude, vLLM, Gemini REST, …):
1. Subclass LLMBackend
2. Implement generate(), health_check(), name
3. Register a default ModelConfig in model_configs.py

The rest of the pipeline — prompt building, JSON parsing, checkpointing,
metric computation — is entirely backend-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """
    Parameters forwarded verbatim to the model's generation endpoint.

    Backends extract what they support; unknown fields are silently ignored,
    so adding a new field here does not break existing backends.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    num_predict: int = 128       # max tokens to generate (Ollama naming; maps to max_tokens elsewhere)
    timeout: int = 120           # per-request wall-clock timeout in seconds
    stop: List[str] = field(default_factory=list)   # early-stop token sequences


class LLMBackend(ABC):
    """
    Abstract base for LLM inference backends.

    Contract:
    - generate() must NEVER raise; return None on any failure.
    - health_check() must return within a few seconds.
    - name is a stable identifier used in log messages and result files.
    """

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """
        Send prompt to the model and return the raw response string.

        Returns None on timeout, network error, or API error.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and the model is loaded."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, e.g. 'ollama/qwen2.5:7b'."""
        ...