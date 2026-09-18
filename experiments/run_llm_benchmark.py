"""
experiments/run_llm_benchmark.py
---------------------------------
Multi-model benchmarking: run the same evaluation across several LLMs and
produce a unified comparison table for the thesis results section.

Addresses RQ3 (prompt engineering) and RQ4 (LLM vs. BERT comparison).

Usage:
    # Compare default models on 500 rows
    python3 experiments/run_llm_benchmark.py --sample 500

    # Custom model list
    python3 experiments/run_llm_benchmark.py \\
        --models qwen2.5:7b qwen3:8b llama3.1:8b \\
        --sample 500 \\
        --prompt-variant joint_fewshot

    # Full test set, all defaults
    python3 experiments/run_llm_benchmark.py

Output (<output-dir>/):
    checkpoint_<model_slug>_<variant>.jsonl   per-model resume checkpoints
    failed_<model_slug>_<variant>.jsonl       parse failures per model
    metrics_<model_slug>.csv                  per-label metrics per model
    comparison.csv                            all models × all labels, full metrics
    summary.json                              macro metrics + timing per model

comparison.csv columns:
    model, prompt_variant, label,
    precision, recall, f1, f2, mcc, s_score, support_pos, support_total,
    n_success, n_fallback, n_total,
    runtime_s, avg_latency_s, p95_latency_s
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
    _FALLBACK,
    LABELS,
    get_model_config,
)
from src.training.transformer_metrics import compute_multilabel_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/final/unified_final_v1.parquet")

DEFAULT_MODELS = [
    "qwen2.5:7b",
    "qwen3:8b",
    "llama3.1:8b",
    "gemma3:12b",
    "mistral-small",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-model LLM benchmarking")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=(
            "Ollama model tags to evaluate. "
            "Each model must be pulled locally (`ollama pull <model>`). "
            f"Default: {DEFAULT_MODELS}"
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="joint_fewshot",
        help="Prompt variant applied to all models (default: joint_fewshot).",
    )
    parser.add_argument(
        "--mode", choices=["joint", "per_label"], default="joint",
        help="Inference mode applied to all models.",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Random subset size (default: full gold test set).")
    parser.add_argument("--output-dir", default="results/llm_benchmark")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Global timeout override (overrides per-model defaults).")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Global retry override.")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="Skip models that fail health check instead of aborting.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    return model.replace(":", "_").replace(".", "").replace("-", "_")


def _load_test_set(sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["split"] == "test") & (df["is_gold"] == True)].reset_index(drop=True)
    logger.info("Gold test rows: %d", len(df))
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", len(df), seed)
    return df


def _run_one_model(
    model: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict] | None:
    """
    Run inference for one model.

    Returns (metrics_df, stats_dict) or None if the model is unavailable
    and --skip-unavailable was set.
    """
    slug = _model_slug(model)
    variant_tag = f"{args.mode}_{args.prompt_variant}"
    checkpoint_path = output_dir / f"checkpoint_{slug}_{variant_tag}.jsonl"
    failed_path = output_dir / f"failed_{slug}_{variant_tag}.jsonl"
    n = len(df)

    backend = OllamaBackend(model=model, base_url=args.ollama_url)
    config = get_model_config(model)

    if args.timeout is not None:
        config.timeout = args.timeout
    if args.max_attempts is not None:
        config.max_attempts = args.max_attempts

    engine = OllamaInferenceEngine(backend, config)

    if not engine.verify_backend():
        if args.skip_unavailable:
            logger.warning("Skipping unavailable model: %s", model)
            return None
        logger.warning(
            "Model '%s' may be unavailable — proceeding with timeout protection.", model
        )

    logger.info(
        "Starting model: %s | adapter: %s | variant: %s | mode: %s",
        backend.name, config.prompt_adapter, args.prompt_variant, args.mode,
    )

    done, stats = engine.predict_batch(
        df=df,
        prompt_variant=args.prompt_variant,
        mode=args.mode,
        checkpoint_path=checkpoint_path,
        failed_path=failed_path,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_true = df[LABELS].values.astype(float)
    label_mask = (~np.isnan(y_true)).astype(float)
    y_true_safe = np.nan_to_num(y_true, nan=0.0)

    preds = np.array(
        [[done.get(i, _FALLBACK)[label] for label in LABELS] for i in range(n)],
        dtype=float,
    )
    fake_logits = preds * 20.0 - 10.0

    metrics_df = compute_multilabel_metrics(
        logits=fake_logits,
        labels=y_true_safe,
        label_mask=label_mask,
        thresholds=None,
        split_name="test",
    )

    print_metrics_table(f"{model} [{args.prompt_variant}]", metrics_df)

    # Save per-model metrics CSV
    metrics_df.to_csv(output_dir / f"metrics_{slug}.csv", index=False)

    return metrics_df, stats.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_test_set(args.sample, args.seed)

    comparison_rows = []
    summary_rows = []

    for model in args.models:
        logger.info("=" * 70)
        logger.info("MODEL: %s", model)

        result = _run_one_model(model, df, args, output_dir)
        if result is None:
            continue

        metrics_df, stats_dict = result
        macro = metrics_df[metrics_df["label"] == "MACRO"].iloc[0]

        for _, row in metrics_df.iterrows():
            is_macro = row["label"] == "MACRO"
            comparison_rows.append({
                "model":           model,
                "prompt_variant":  args.prompt_variant,
                "mode":            args.mode,
                "label":           row["label"],
                "precision":       round(float(row["precision"]), 4),
                "recall":          round(float(row["recall"]), 4),
                "f1":              round(float(row["f1"]), 4),
                "f2":              round(float(row["f2"]), 4),
                "mcc":             round(float(row["mcc"]), 4),
                "s_score":         round(float(row["s_score"]), 4),
                "support_pos":     int(row["support_pos"]),
                "support_total":   int(row["support_total"]),
                # Timing + counts on MACRO row only (avoids redundant repetition)
                "n_success":       stats_dict["n_success"] if is_macro else "",
                "n_fallback":      stats_dict["n_fallback"] if is_macro else "",
                "n_total":         stats_dict["n_total"] if is_macro else "",
                "runtime_s":       stats_dict["total_runtime_s"] if is_macro else "",
                "avg_latency_s":   stats_dict["avg_latency_s"] if is_macro else "",
                "p95_latency_s":   stats_dict["p95_latency_s"] if is_macro else "",
            })

        summary_rows.append({
            "model":           model,
            "prompt_variant":  args.prompt_variant,
            "macro_f1":        round(float(macro["f1"]), 4),
            "macro_f2":        round(float(macro["f2"]), 4),
            "macro_s_score":   round(float(macro["s_score"]), 4),
            "macro_mcc":       round(float(macro["mcc"]), 4),
            **stats_dict,
        })

    if not comparison_rows:
        logger.error("No models produced results. Exiting.")
        sys.exit(1)

    # ── Save comparison.csv ────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    logger.info("Saved comparison.csv → %s", output_dir / "comparison.csv")

    # ── Save summary.json ──────────────────────────────────────────────────────
    summary = {
        "prompt_variant": args.prompt_variant,
        "mode":           args.mode,
        "n_rows":         len(df),
        "seed":           args.seed,
        "models":         summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── Print summary table ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    display_cols = [
        "macro_s_score", "macro_f1", "macro_f2", "macro_mcc",
        "n_success", "n_fallback", "avg_latency_s", "total_runtime_s",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]

    print("\n" + "=" * 76)
    print(
        f"BENCHMARK SUMMARY — variant={args.prompt_variant} "
        f"mode={args.mode} n={len(df)}"
    )
    print("=" * 76)
    print(summary_df[display_cols].to_string())
    print("=" * 76)

    if summary_rows:
        best = summary_df["macro_s_score"].idxmax()
        logger.info(
            "Best model by Macro S-Score: %s (%.4f)",
            best, summary_df.loc[best, "macro_s_score"],
        )

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()