"""
experiments/run_ollama_eval.py
-------------------------------
CLI wrapper for single-model, single-variant Ollama evaluation.

All inference logic lives in src/llm/. This script handles argument parsing,
data loading, metric computation, and result serialisation.

Usage:
    # Full gold test set (≈13 250 rows)
    python3 experiments/run_ollama_eval.py

    # Quick smoke test on 10 rows
    python3 experiments/run_ollama_eval.py --sample 10

    # Prompt variant ablation
    python3 experiments/run_ollama_eval.py \\
        --model qwen3:8b \\
        --prompt-variant joint_fewshot \\
        --sample 500 \\
        --output-dir results/qwen3_fewshot

    # Per-label mode (Ma et al. 2025, avoids label-suppression)
    python3 experiments/run_ollama_eval.py --mode per_label

Output:
    <output-dir>/
        checkpoint_<mode>_<variant>.jsonl   resume-safe row predictions
        failed_responses.jsonl              raw responses that failed to parse
        test_metrics.csv                    per-label metrics table
        summary.json                        macro metrics + run metadata
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import (
    OllamaBackend,
    OllamaInferenceEngine,
    PROMPT_VARIANTS,
    get_model_config,
    _FALLBACK,
    LABELS,
)
from src.training.transformer_metrics import compute_multilabel_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/final/unified_final_v1.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama multilabel evaluation")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--mode", choices=["joint", "per_label"], default="joint",
        help="joint: 1 call/row (JSON); per_label: 4 calls/row (binary, Ma et al. 2025)",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="joint_basic",
        help="Prompt template variant (joint mode only).",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--output-dir", default=None,
        help="Default: results/ollama_<model>_<mode>_<variant>",
    )
    parser.add_argument("--timeout", type=int, default=None,
                        help="Override per-request timeout (default: from MODEL_CONFIGS)")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Override retry attempts (default: from MODEL_CONFIGS)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    return model.replace(":", "_").replace(".", "").replace("-", "_")


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    slug = _model_slug(args.model)
    return Path(f"results/ollama_{slug}_{args.mode}_{args.prompt_variant}")


def _load_test_set(sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["split"] == "test") & (df["is_gold"] == True)].reset_index(drop=True)
    logger.info("Gold test rows: %d", len(df))
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", len(df), seed)
    return df


def _compute_and_save_metrics(
    df: pd.DataFrame,
    done: dict,
    output_dir: Path,
    args: argparse.Namespace,
    stats,
) -> None:
    n = len(df)
    y_true = df[LABELS].values.astype(float)
    label_mask = (~np.isnan(y_true)).astype(float)
    y_true_safe = np.nan_to_num(y_true, nan=0.0)

    preds = np.array(
        [[done.get(i, _FALLBACK)[label] for label in LABELS] for i in range(n)],
        dtype=float,
    )
    # sigmoid(-10) ≈ 0 → pred=0;  sigmoid(+10) ≈ 1 → pred=1
    fake_logits = preds * 20.0 - 10.0

    metrics_df = compute_multilabel_metrics(
        logits=fake_logits,
        labels=y_true_safe,
        label_mask=label_mask,
        thresholds=None,
        split_name="test",
    )

    title = (
        f"OLLAMA {args.model} [{args.prompt_variant}] "
        f"mode={args.mode} — TEST METRICS"
    )
    print_metrics_table(title, metrics_df)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    macro = metrics_df[metrics_df["label"] == "MACRO"].iloc[0]
    summary = {
        "model": args.model,
        "mode": args.mode,
        "prompt_variant": args.prompt_variant,
        "ollama_url": args.ollama_url,
        "n_total": n,
        "n_success": stats.n_success + stats.n_resumed,
        "n_fallback": stats.n_fallback,
        "avg_latency_s": round(stats.avg_latency_s, 3),
        "total_runtime_s": round(stats.total_runtime_s, 1),
        "macro_f1":      round(float(macro["f1"]), 4),
        "macro_f2":      round(float(macro["f2"]), 4),
        "macro_s_score": round(float(macro["s_score"]), 4),
        "macro_mcc":     round(float(macro["mcc"]), 4),
        "per_label": {
            row["label"]: {
                "precision":     round(float(row["precision"]), 4),
                "recall":        round(float(row["recall"]), 4),
                "f1":            round(float(row["f1"]), 4),
                "f2":            round(float(row["f2"]), 4),
                "s_score":       round(float(row["s_score"]), 4),
                "support_pos":   int(row["support_pos"]),
                "support_total": int(row["support_total"]),
            }
            for _, row in metrics_df[metrics_df["label"] != "MACRO"].iterrows()
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    logger.info(
        "Macro S-Score=%.4f | F1=%.4f | F2=%.4f | MCC=%.4f",
        summary["macro_s_score"],
        summary["macro_f1"],
        summary["macro_f2"],
        summary["macro_mcc"],
    )
    logger.info("Results saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = _build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_{args.mode}_{args.prompt_variant}.jsonl"
    failed_path = output_dir / "failed_responses.jsonl"

    df = _load_test_set(args.sample, args.seed)

    # ── Build engine ───────────────────────────────────────────────────────────
    backend = OllamaBackend(model=args.model, base_url=args.ollama_url)
    config = get_model_config(args.model)

    # CLI overrides take precedence over registry defaults
    if args.timeout is not None:
        config.timeout = args.timeout
    if args.max_attempts is not None:
        config.max_attempts = args.max_attempts

    engine = OllamaInferenceEngine(backend, config)

    if not engine.verify_backend():
        logger.warning("Backend unreachable; proceeding anyway (may time out).")

    logger.info(
        "Backend: %s | adapter: %s | variant: %s | mode: %s",
        backend.name, config.prompt_adapter, args.prompt_variant, args.mode,
    )

    # ── Inference ──────────────────────────────────────────────────────────────
    done, stats = engine.predict_batch(
        df=df,
        prompt_variant=args.prompt_variant,
        mode=args.mode,
        checkpoint_path=checkpoint_path,
        failed_path=failed_path,
    )

    # ── Metrics & save ─────────────────────────────────────────────────────────
    _compute_and_save_metrics(df, done, output_dir, args, stats)


if __name__ == "__main__":
    main()
