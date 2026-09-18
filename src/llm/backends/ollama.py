"""
src/llm/backends/ollama.py
--------------------------
OllamaBackend: calls the Ollama /api/generate endpoint.

Supports all models available via `ollama pull`, including:
    qwen2.5:7b, qwen3:8b, llama3.1:8b, gemma3:12b,
    mistral-small, deepseek-r1, and any future pulls.

Usage:
    backend = OllamaBackend(model="qwen2.5:7b")
    raw = backend.generate(prompt, config)
"""

import logging
from typing import Optional

import requests

from .base import GenerationConfig, LLMBackend

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """
    Stateless HTTP client for the Ollama /api/generate endpoint.

    Each call is independent — no session state is held between calls.
    Thread-safe as long as the underlying requests library is (it is).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")

    # ── LLMBackend interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        options: dict = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.num_predict,
        }
        if config.stop:
            options["stop"] = config.stop

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=config.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.warning(
                "Ollama timeout after %ds [model=%s]", config.timeout, self._model
            )
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama connection refused — is Ollama running at %s?", self.base_url
            )
            return None
        except Exception as exc:
            logger.warning("Ollama request failed [model=%s]: %s", self._model, exc)
            return None

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Convenience ───────────────────────────────────────────────────────────

    def list_local_models(self) -> list:
        """Return list of model tags currently pulled in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
