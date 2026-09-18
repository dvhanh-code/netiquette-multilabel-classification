"""
src/llm/gemini_client.py
------------------------
GeminiClient using google.genai SDK (new, replaces google.generativeai).

Install:
    pip install google-genai

Model: gemini-2.5-flash (free tier available)
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        gap = self.min_interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.monotonic()


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        requests_per_minute: int = 10,
        max_retries: int = 5,
        temperature: float = 0.0,
        cache_path: Optional[str] = None,
    ):
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError as e:
            raise ImportError(
                "google-genai is required. "
                "Install with: pip install google-genai"
            ) from e

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=resolved_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._rate_limiter = RateLimiter(requests_per_minute)

        # Cache
        self._cache: dict = {}
        self._cache_path: Optional[Path] = None
        if cache_path:
            self._cache_path = Path(cache_path)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _load_cache(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        with self._cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._cache[entry["input_hash"]] = entry["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Loaded %d cached results from %s",
                    len(self._cache), self._cache_path)

    def _write_to_cache(self, input_hash: str, system: str,
                        user: str, response: str) -> None:
        if not self._cache_path:
            return
        entry = {
            "input_hash": input_hash,
            "model": self.model_name,
            "system_snippet": system[:80],
            "user_snippet": user[:80],
            "response": response,
        }
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _make_hash(self, system_prompt: str, user_prompt: str) -> str:
        combined = system_prompt + "\n|||SPLIT|||\n" + user_prompt
        return _text_hash(combined)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        h = self._make_hash(system_prompt, user_prompt)

        if h in self._cache:
            return self._cache[h]

        full_prompt = system_prompt + "\n\n" + user_prompt

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.wait()

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=500,
                        thinking_config=self._types.ThinkingConfig(
                            thinking_budget=0),
                        safety_settings=[
                            self._types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_NONE"),
                            self._types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_NONE"),
                            self._types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_NONE"),
                            self._types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE"),
                        ]
                    )
                )

                text = response.text
                if text is None:
                    logger.warning("Response blocked by safety filter, returning empty")
                    return "{}"

                text = text.strip()
                self._cache[h] = text
                self._write_to_cache(h, system_prompt, user_prompt, text)
                return text

            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) * 2
                error_str = str(e).lower()

                if any(x in error_str for x in
                       ["quota", "429", "rate", "resource_exhausted"]):
                    logger.warning(
                        "Rate limit (attempt %d/%d). Waiting %ds.",
                        attempt + 1, self.max_retries, wait_time)
                    time.sleep(wait_time)
                elif any(x in error_str for x in ["500", "503", "unavailable"]):
                    logger.warning(
                        "Server error (attempt %d/%d). Waiting %ds.",
                        attempt + 1, self.max_retries, wait_time)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Gemini API error: {e}") from e

        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def cache_size(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model={self.model_name!r}, "
            f"cache_size={self.cache_size()})"
        )
