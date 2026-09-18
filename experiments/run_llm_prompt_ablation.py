"""
experiments/run_llm_prompt_ablation.py
---------------------------------------
Prompt variant ablation: run multiple prompt templates on the same sample
and compare the resulting metrics side-by-side.

Answers RQ3: which prompt engineering strategy best closes the gap between
naive zero-shot and a definition-rich / few-shot / self-check approach?

Usage:
    # All 5 variants on 500 rows
    python3 experiments/run_llm_prompt_ablation.py --sample 500

    # Custom model and variants
    python3 experiments/run_llm_prompt_ablation.py \\
        --model qwen3:8b \\
        --sample 500 \\
        --variants joint_basic joint_definitions joint_fewshot

    # Full test set — may take several hours
    python3 experiments/run_llm_prompt_ablation.py --model llama3.1:8b

Output (<output-dir>/):
    checkpoint_<variant>.jsonl      per-variant resume checkpoint
    failed_<variant>.jsonl          failed parse responses per variant
    comparison.csv                  all variants × all labels, full metrics
    summary.json                    macro metrics per variant + timing

comparison.csv columns:
    variant, label, precision, recall, f1, f2, mcc, s_score,
    support_pos, support_total, n_success, n_fallback, runtime_s, avg_latency_s
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
DEFAULT_VARIANTS = ["joint_basic", "joint_definitions", "joint_fewshot", "joint_rules", "joint_selfcheck"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM prompt variant ablation")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--variants", nargs="+",
        choices=sorted(PROMPT_VARIANTS),
        default=DEFAULT_VARIANTS,
        help="Prompt variants to compare (default: all 5).",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-dir", default="results/llm_prompt_ablation")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant: str,
    df: pd.DataFrame,
    engine: OllamaInferenceEngine,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Run inference for one prompt variant.

    Returns:
        metrics_df: output of compute_multilabel_metrics
        stats_dict: timing and success/fallback counts
    """
    checkpoint_path = output_dir / f"checkpoint_{variant}.jsonl"
    failed_path = output_dir / f"failed_{variant}.jsonl"
    n = len(df)

    done, stats = engine.predict_batch(
        df=df,
        prompt_variant=variant,
        mode="joint",
        checkpoint_path=checkpoint_path,
        failed_path=failed_path,
    )

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

    print_metrics_table(f"Variant: {variant}", metrics_df)
    return metrics_df, stats.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading gold test set from %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["split"] == "test") & (df["is_gold"] == True)].reset_index(drop=True)
    logger.info("Gold test rows: %d", len(df))

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", len(df), args.seed)

    # ── Build engine (shared across all variants) ──────────────────────────────
    backend = OllamaBackend(model=args.model, base_url=args.ollama_url)
    config = get_model_config(args.model)
    if args.timeout is not None:
        config.timeout = args.timeout
    if args.max_attempts is not None:
        config.max_attempts = args.max_attempts

    engine = OllamaInferenceEngine(backend, config)

    if not engine.verify_backend():
        logger.warning("Backend unreachable; proceeding anyway.")

    logger.info(
        "Model: %s | adapter: %s | variants: %s",
        backend.name, config.prompt_adapter, args.variants,
    )

    # ── Run ablation ───────────────────────────────────────────────────────────
    comparison_rows = []
    summary_rows = []

    for variant in args.variants:
        logger.info("=" * 60)
        logger.info("Starting variant: %s", variant)
        metrics_df, stats_dict = run_variant(variant, df, engine, output_dir)

        macro = metrics_df[metrics_df["label"] == "MACRO"].iloc[0]

        for _, row in metrics_df.iterrows():
            is_macro = row["label"] == "MACRO"
            comparison_rows.append({
                "variant":       variant,
                "label":         row["label"],
                "precision":     round(float(row["precision"]), 4),
                "recall":        round(float(row["recall"]), 4),
                "f1":            round(float(row["f1"]), 4),
                "f2":            round(float(row["f2"]), 4),
                "mcc":           round(float(row["mcc"]), 4),
                "s_score":       round(float(row["s_score"]), 4),
                "support_pos":   int(row["support_pos"]),
                "support_total": int(row["support_total"]),
                # Runtime columns only on MACRO row
                "n_success":     stats_dict["n_success"] if is_macro else "",
                "n_fallback":    stats_dict["n_fallback"] if is_macro else "",
                "runtime_s":     stats_dict["total_runtime_s"] if is_macro else "",
                "avg_latency_s": stats_dict["avg_latency_s"] if is_macro else "",
            })

        summary_rows.append({
            "variant":       variant,
            "macro_f1":      round(float(macro["f1"]), 4),
            "macro_f2":      round(float(macro["f2"]), 4),
            "macro_s_score": round(float(macro["s_score"]), 4),
            "macro_mcc":     round(float(macro["mcc"]), 4),
            **stats_dict,
        })

    # ── Save outputs ───────────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    logger.info("Saved comparison.csv → %s", output_dir / "comparison.csv")

    summary = {
        "model": args.model,
        "n_rows": len(df),
        "seed": args.seed,
        "variants": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── Print macro comparison table ───────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).set_index("variant")
    display_cols = ["macro_s_score", "macro_f1", "macro_f2", "macro_mcc",
                    "n_success", "n_fallback", "avg_latency_s"]
    display_cols = [c for c in display_cols if c in summary_df.columns]

    print("\n" + "=" * 72)
    print(f"ABLATION SUMMARY — {args.model} — {len(df)} rows")
    print("=" * 72)
    print(summary_df[display_cols].to_string())
    print("=" * 72)

    best = summary_df["macro_s_score"].idxmax()
    logger.info(
        "Best variant by Macro S-Score: %s (%.4f)",
        best, summary_df.loc[best, "macro_s_score"],
    )
    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
