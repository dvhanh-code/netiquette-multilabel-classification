"""
src/llm/engine.py
-----------------
OllamaInferenceEngine: orchestrates prompt building → API call → parsing
→ retry → fallback → checkpointing for any LLMBackend.

Key properties
--------------
- Backend-agnostic: works with OllamaBackend today; swap in OpenAIBackend,
  ClaudeBackend, etc. tomorrow without touching this file.
- Resume-safe: every completed row is appended to a checkpoint JSONL file
  immediately after inference, so interrupted runs resume at zero cost.
- Latency-aware: each call is timed; BatchStats aggregates per-run timing
  for benchmarking (comparison.csv runtime_s and avg_latency_s columns).
- Failure-transparent: rows that exhaust all retries are logged to a
  separate failed_responses.jsonl file for qualitative error analysis.
- Mode-flexible: supports 'joint' (one API call, JSON output) and
  'per_label' (4 calls, binary output) within the same engine instance.

Usage:
    from src.llm.backends import OllamaBackend
    from src.llm.model_configs import get_model_config
    from src.llm.engine import OllamaInferenceEngine

    backend = OllamaBackend(model="qwen3:8b")
    config  = get_model_config("qwen3:8b")
    engine  = OllamaInferenceEngine(backend, config)

    done, stats = engine.predict_batch(
        df, prompt_variant="joint_fewshot",
        checkpoint_path=Path("results/checkpoint.jsonl"),
        failed_path=Path("results/failed.jsonl"),
    )
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backends.base import GenerationConfig, LLMBackend
from .checkpoint import append_checkpoint, load_checkpoint
from .model_configs import ModelConfig
from .parsing import LABELS, RobustJSONParser, _FALLBACK, parse_binary
from .prompt_registry import build_per_label_prompt, build_prompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Single-row inference result."""
    predictions: Dict[str, int]
    success: bool
    latency_s: float
    raw_response: Optional[str] = None


@dataclass
class BatchStats:
    """Aggregate statistics for a predict_batch() call."""
    n_total: int = 0
    n_success: int = 0
    n_fallback: int = 0
    n_resumed: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency_s(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def p95_latency_s(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    @property
    def total_runtime_s(self) -> float:
        return float(np.sum(self.latencies)) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_fallback": self.n_fallback,
            "n_resumed": self.n_resumed,
            "avg_latency_s": round(self.avg_latency_s, 3),
            "p95_latency_s": round(self.p95_latency_s, 3),
            "total_runtime_s": round(self.total_runtime_s, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class OllamaInferenceEngine:
    """
    High-level inference orchestrator for Ollama-compatible LLM backends.

    Args:
        backend:      Any LLMBackend implementation (OllamaBackend, etc.).
        model_config: ModelConfig providing generation params and prompt adapter.
        parser:       RobustJSONParser instance (shared or custom).
        log_every:    Log progress every N newly completed rows.
    """

    def __init__(
        self,
        backend: LLMBackend,
        model_config: ModelConfig,
        parser: Optional[RobustJSONParser] = None,
        log_every: int = 100,
    ) -> None:
        self.backend = backend
        self.config = model_config
        self.parser = parser or RobustJSONParser()
        self.log_every = log_every

        # Build the GenerationConfig once; all predict() calls share it
        self._gen_config = GenerationConfig(
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            num_predict=model_config.num_predict,
            timeout=model_config.timeout,
            stop=model_config.stop,
        )

    # ── Single-row prediction ──────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        prompt_variant: str = "joint_basic",
        failed_file: Optional[IO] = None,
    ) -> InferenceResult:
        """
        Predict labels for one text with retry logic.

        Retry loop:
          - Calls backend.generate() up to config.max_attempts times.
          - After each successful API call, attempts to parse the response.
          - On parse success: returns immediately.
          - On all-fail: writes raw responses to failed_file (if provided),
            returns all-zeros fallback, success=False.

        Args:
            text:           German comment text to classify.
            prompt_variant: Key into prompt_registry.PROMPTS.
            failed_file:    Open file handle for failed-response logging.

        Returns:
            InferenceResult with predictions, success flag, and wall-clock latency.
        """
        prompt = build_prompt(text, prompt_variant, self.config.prompt_adapter)
        raw_responses: List[str] = []
        t_start = time.perf_counter()

        for attempt in range(self.config.max_attempts):
            raw = self.backend.generate(prompt, self._gen_config)
            if raw is None:
                logger.debug(
                    "[%s] Attempt %d/%d: no response from backend",
                    self.backend.name, attempt + 1, self.config.max_attempts,
                )
                continue

            raw_responses.append(raw)
            pred = self.parser.parse(raw)
            if pred is not None:
                return InferenceResult(
                    predictions=pred,
                    success=True,
                    latency_s=time.perf_counter() - t_start,
                    raw_response=raw,
                )

            logger.debug(
                "[%s] Parse fail attempt %d/%d: %.80r",
                self.backend.name, attempt + 1, self.config.max_attempts, raw,
            )

        # All attempts exhausted
        if failed_file is not None:
            entry = {
                "backend": self.backend.name,
                "prompt_variant": prompt_variant,
                "text_preview": text[:300],
                "raw_responses": raw_responses,
            }
            failed_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            failed_file.flush()

        return InferenceResult(
            predictions=dict(_FALLBACK),
            success=False,
            latency_s=time.perf_counter() - t_start,
            raw_response=raw_responses[-1] if raw_responses else None,
        )

    def predict_per_label(self, text: str) -> InferenceResult:
        """
        Make 4 independent binary API calls, one per label.

        Avoids the label-suppression bias of joint JSON generation documented
        in Ma et al. (2025, EMNLP): autoregressive models assign lower probability
        to later labels once earlier labels dominate the prefix distribution.
        Each independent call removes that cross-label interference.

        Returns:
            InferenceResult where success=True always (binary fallback = 0 per label).
        """
        t_start = time.perf_counter()
        predictions: Dict[str, int] = {}

        for label in LABELS:
            prompt = build_per_label_prompt(text, label)
            raw = self.backend.generate(prompt, self._gen_config) or ""
            predictions[label] = parse_binary(raw)

        return InferenceResult(
            predictions=predictions,
            success=True,
            latency_s=time.perf_counter() - t_start,
        )

    # ── Batch prediction ───────────────────────────────────────────────────────

    def predict_batch(
        self,
        df: pd.DataFrame,
        prompt_variant: str = "joint_basic",
        mode: str = "joint",
        checkpoint_path: Optional[Path] = None,
        failed_path: Optional[Path] = None,
    ) -> Tuple[Dict[int, Dict[str, int]], BatchStats]:
        """
        Run inference over an entire DataFrame with checkpointing and timing.

        Args:
            df:               DataFrame with a 'text' column.
            prompt_variant:   Prompt variant key (ignored when mode='per_label').
            mode:             'joint' or 'per_label'.
            checkpoint_path:  JSONL file for resume-safe checkpointing.
            failed_path:      JSONL file for logging failed parse responses.

        Returns:
            done:  {row_idx → {label: 0|1}} for all rows (resumed + newly inferred).
            stats: BatchStats with latency tracking and counts.

        Resume behavior:
            Rows already present in checkpoint_path are skipped; their predictions
            are loaded into done but not counted in BatchStats.latencies.
        """
        done = load_checkpoint(checkpoint_path) if checkpoint_path else {}
        n = len(df)
        stats = BatchStats(n_total=n, n_resumed=len(done))

        if stats.n_resumed > 0:
            logger.info(
                "[%s] Resuming — %d/%d rows already in checkpoint",
                self.backend.name, stats.n_resumed, n,
            )

        failed_file: Optional[IO] = None
        if failed_path is not None:
            failed_file = open(failed_path, "a", encoding="utf-8")

        try:
            for i in range(n):
                if i in done:
                    continue

                text = str(df.at[i, "text"])

                if mode == "per_label":
                    result = self.predict_per_label(text)
                else:
                    result = self.predict(
                        text=text,
                        prompt_variant=prompt_variant,
                        failed_file=failed_file,
                    )

                done[i] = result.predictions
                stats.latencies.append(result.latency_s)

                if result.success:
                    stats.n_success += 1
                else:
                    stats.n_fallback += 1

                if checkpoint_path is not None:
                    append_checkpoint(
                        checkpoint_path, i, result.predictions, result.success
                    )

                new_done = stats.n_resumed + stats.n_success + stats.n_fallback
                if new_done % self.log_every == 0 or new_done == n:
                    logger.info(
                        "[%s] %d/%d | success=%d | fallback=%d | avg_latency=%.2fs",
                        self.backend.name,
                        new_done, n,
                        stats.n_success, stats.n_fallback,
                        stats.avg_latency_s,
                    )
        finally:
            if failed_file is not None:
                failed_file.close()

        logger.info(
            "[%s] Batch done — total=%d | success=%d | fallback=%d | "
            "runtime=%.1fs | avg_latency=%.2fs",
            self.backend.name, n,
            stats.n_success + stats.n_resumed, stats.n_fallback,
            stats.total_runtime_s, stats.avg_latency_s,
        )

        return done, stats

    # ── Utilities ─────────────────────────────────────────────────────────────

    def verify_backend(self) -> bool:
        """
        Check that the backend is reachable before starting a long batch run.
        Logs a warning (not an exception) if unavailable.
        """
        ok = self.backend.health_check()
        if not ok:
            logger.warning(
                "Backend health check failed for '%s'. "
                "Ensure the service is running before starting inference.",
                self.backend.name,
            )
        return ok

    @staticmethod
    def predictions_to_arrays(
        done: Dict[int, Dict[str, int]],
        n: int,
        fallback: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert a {row_idx → pred_dict} mapping to a (n, 4) numpy array.

        Rows missing from done (should not happen in normal flow) are filled
        with fallback (default: all-zeros).

        Returns:
            Binary float array shaped (n, len(LABELS)) suitable for
            compute_multilabel_metrics via fake-logit conversion.
        """
        fb = fallback or dict(_FALLBACK)
        arr = np.array(
            [[done.get(i, fb)[label] for label in LABELS] for i in range(n)],
            dtype=float,
        )
        return arr
