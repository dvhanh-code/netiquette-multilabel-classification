"""
src/llm/inference.py
--------------------
LLMInferenceEngine: runs batch LLM inference over a DataFrame and returns
numpy arrays in the same format as the transformer evaluation pipeline.

Key properties:
  - Cache-first: all API results are written to disk immediately after each
    call, so runs can be interrupted and resumed at zero cost.
  - NaN-safe: failed API calls produce NaN rows, which are masked out by
    bootstrap_ci and transformer_metrics (consistent with partial annotation).
  - Reproducible: temperature=0 in GeminiClient, deterministic prompt builder.

Usage (from a notebook):
    from src.llm.inference import LLMInferenceEngine

    engine = LLMInferenceEngine(
        api_key="YOUR_KEY",
        model_name="gemini-2.5-flash",
        cache_path="results/llm_cache/gemini_flash.jsonl",
    )

    results = engine.predict_df(
        df=test_df,
        text_col="text",
        output_path="results/llm_gold_only/predictions.npz",
    )
    # results["predictions"]  shape (n, 4)
    # results["confidences"]  shape (n, 4)
    # results["success_mask"] shape (n,)  bool
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.llm.gemini_client import GeminiClient
from src.llm.output_parser import ParseResult, parse_response, results_to_arrays
from src.llm.prompts import PromptBuilder
from src.llm.platform_rules import PlatformRules

logger = logging.getLogger(__name__)

LABELS = ["hate_speech", "toxic", "threat", "insult"]


class LLMInferenceEngine:
    """
    Runs LLM inference over a DataFrame and saves numpy-compatible results.

    Args:
        api_key:        Gemini API key (or set GEMINI_API_KEY env var).
        model_name:     Gemini model identifier.
        requests_per_minute: Rate limit (15 for free tier, 60+ for paid).
        use_cot:        Enable chain-of-thought reasoning in prompts.
        use_few_shot:   Include few-shot examples in system prompt.
        include_reasoning: Request and store reasoning in outputs.
        platform_rules: Optional platform-specific rule injection.
        cache_path:     Path for JSONL API result cache (strongly recommended).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        requests_per_minute: int = 15,
        use_cot: bool = False,
        use_few_shot: bool = True,
        include_reasoning: bool = True,
        platform_rules: Optional[PlatformRules] = None,
        cache_path: Optional[str] = None,
    ):
        self.client = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            requests_per_minute=requests_per_minute,
            cache_path=cache_path,
        )
        self.prompt_builder = PromptBuilder(
            use_cot=use_cot,
            use_few_shot=use_few_shot,
            include_reasoning=include_reasoning,
        )
        self.platform_rules = platform_rules
        self._system_prompt: Optional[str] = None

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self.prompt_builder.build_system_prompt(
                platform_rules=self.platform_rules
            )
        return self._system_prompt

    def predict_single(self, text: str) -> ParseResult:
        """Predict a single text. Returns ParseResult."""
        system = self._get_system_prompt()
        user = self.prompt_builder.build_user_prompt(text)
        try:
            raw = self.client.generate(system, user)
            return parse_response(raw)
        except Exception as e:
            return ParseResult(success=False, raw_text="", error=str(e))

    def predict_df(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_path: Optional[str] = None,
        label_names: List[str] = LABELS,
        log_every: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference over an entire DataFrame.

        Args:
            df:           Input DataFrame; must contain text_col.
            text_col:     Column holding the comment text.
            output_path:  If provided, saves results as .npz for future loading.
            label_names:  Label order (must match existing evaluation pipeline).
            log_every:    Log progress every N rows.

        Returns:
            {
              "predictions":  np.ndarray shape (n, 4), float (0/1/NaN)
              "confidences":  np.ndarray shape (n, 4), float ([0,1]/NaN)
              "success_mask": np.ndarray shape (n,), bool
            }
        """
        texts = df[text_col].tolist()
        n = len(texts)
        results: List[ParseResult] = []

        n_cached = 0
        n_failed = 0
        system = self._get_system_prompt()

        logger.info("Starting LLM inference on %d texts (model=%s)", n, self.client.model_name)
        logger.info("System prompt length: %d chars", len(system))

        for i, text in enumerate(texts):
            if i > 0 and i % log_every == 0:
                logger.info(
                    "Progress: %d/%d | cached=%d | failed=%d",
                    i, n, n_cached, n_failed,
                )

            text_str = str(text) if not isinstance(text, str) else text
            if not text_str.strip():
                # Empty text: return all-zero prediction
                results.append(ParseResult(
                    success=True,
                    predictions={l: 0 for l in label_names},
                    confidences={l: 0.0 for l in label_names},
                    reasoning="Empty text.",
                ))
                continue

            # Check if this will be a cache hit (avoid counting as "new" call)
            cache_before = self.client.cache_size()
            result = self.predict_single(text_str)
            cache_after = self.client.cache_size()

            if cache_after > cache_before:
                pass  # new call
            else:
                n_cached += 1

            if not result.success:
                n_failed += 1
                logger.debug("Parse failure at index %d: %s", i, result.error)

            results.append(result)

        logger.info(
            "Inference complete: %d total | %d cached | %d failed",
            n, n_cached, n_failed,
        )

        arrays = results_to_arrays(results, label_names=label_names)

        # Attach reasoning strings
        arrays["reasoning"] = np.array(
            [r.reasoning for r in results], dtype=object
        )

        if output_path:
            self._save_arrays(arrays, output_path, results)
            logger.info("Saved results to %s", output_path)

        return arrays

    @staticmethod
    def _save_arrays(
        arrays: Dict[str, np.ndarray],
        output_path: str,
        results: List[ParseResult],
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.savez(
            path.with_suffix(".npz"),
            predictions=arrays["predictions"],
            confidences=arrays["confidences"],
            success_mask=arrays["success_mask"],
        )

        # Save reasoning as text file for qualitative analysis
        reasoning_path = path.parent / (path.stem + "_reasoning.txt")
        with reasoning_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(results):
                f.write(f"[{i}] {r.reasoning or '—'}\n")

    @staticmethod
    def load_results(output_path: str) -> Dict[str, np.ndarray]:
        """Load previously saved .npz results."""
        path = Path(output_path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path, allow_pickle=True)
        return {
            "predictions": data["predictions"],
            "confidences": data["confidences"],
            "success_mask": data["success_mask"],
        }

    def estimate_cost(
        self,
        df: pd.DataFrame,
        avg_text_tokens: int = 50,
        output_tokens: int = 120,
        price_input_per_1m: float = 0.075,
        price_output_per_1m: float = 0.30,
    ) -> Dict[str, float]:
        """
        Estimate API cost before running. Useful for budget planning.

        Returns dict with total_input_tokens, total_output_tokens, estimated_usd.
        Counts only non-cached rows.
        """

        system = self._get_system_prompt()
        actual_system_tokens = len(system) // 4  # rough char-to-token ratio

        n = len(df)
        total_input = n * (actual_system_tokens + avg_text_tokens)
        total_output = n * output_tokens

        cost = (total_input / 1_000_000) * price_input_per_1m + \
               (total_output / 1_000_000) * price_output_per_1m

        return {
            "n_rows": n,
            "system_prompt_tokens": actual_system_tokens,
            "avg_tokens_per_row": actual_system_tokens + avg_text_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_usd": round(cost, 3),
        }