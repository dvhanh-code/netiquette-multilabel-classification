"""
experiments/run_llm_eval.py
----------------------------
Run LLM-only evaluation on the test set and produce bootstrap CI results.

Usage:
    # Run on full test set (free tier: ~15 min, ~$0.80 with Gemini Flash)
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY

    # Run on a stratified sample (faster, for initial experiments)
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY --sample 500

    # Use a different model
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY --model gemini-1.5-pro

    # Enable chain-of-thought (better accuracy, ~30% more tokens)
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY --cot

    # Apply platform rules
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY --platform news_comments

    # Dry-run: estimate cost without making API calls
    python3 experiments/run_llm_eval.py --api-key YOUR_KEY --dry-run

Output directory (--output-dir):
    predictions.npz          numpy arrays (predictions, confidences, success_mask)
    predictions_reasoning.txt  per-row reasoning strings
    bootstrap_ci_report.csv  per-label and macro CI results
    summary.json             key metrics for thesis table
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.inference import LLMInferenceEngine
from src.llm.platform_rules import get_platform_rules, list_presets
from src.evaluation.bootstrap_ci import bootstrap_ci_report, print_ci_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/final/unified_final_v1.parquet")
LABELS = ["hate_speech", "toxic", "threat", "insult"]


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-only multilabel evaluation")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model name")
    parser.add_argument("--rpm", type=int, default=15,
                        help="Requests per minute (15 for free tier, 60+ for paid)")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to evaluate on")
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratified sample size (default: full split)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/llm_{model}_{split})")
    parser.add_argument("--cache-path", default=None,
                        help="JSONL cache path (default: output_dir/cache.jsonl)")
    parser.add_argument("--platform", default=None, choices=list_presets() + [None],
                        help=f"Platform preset: {list_presets()}")
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought reasoning in prompts")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Disable few-shot examples (ablation)")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Omit reasoning field from JSON output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without making API calls")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


"""
def load_split(split: str, sample: int = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["split"] == split].reset_index(drop=True)

    # Evaluation always on gold rows only
    df = df[df["is_gold"] == True].reset_index(drop=True)

    logger.info("Loaded %d gold %s rows", len(df), split)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows", len(df))

    return df
"""

def load_split(split: str, sample: int = None, seed: int = 42,
                min_label_positives: int = 20) -> pd.DataFrame:
    """
    Load a data split, optionally taking a STRATIFIED sample that guarantees
    every label — especially rare ones like 'threat' — has positive examples.

    Logic:
        1. Load all gold rows for the split (unchanged from before).
        2. If `sample` is set and smaller than the full split:
           a. For each label, keep up to `min_label_positives` positive rows
              (or ALL positives if the label has fewer than that available —
              this is exactly what happens for threat, which typically has
              far fewer than 20 positives in val/test).
           b. Union these "must-keep" rows across all 4 labels (deduplicated).
           c. Fill the remaining budget with a random sample from the rest,
              same as the original behavior.
        3. Shuffle the final result so must-keep rows aren't clustered at
           the top (avoids any ordering bias in downstream processing).

    Args:
        min_label_positives: minimum positives to guarantee per label.
            Default 20 — safely above threat's ~21 total, so in practice
            ALL threat positives get included whenever threat has <20 rows.
    """
    df = pd.read_parquet(DATA_PATH)
    df = df[df["split"] == split].reset_index(drop=True)

    # Evaluation always on gold rows only
    df = df[df["is_gold"] == True].reset_index(drop=True)

    logger.info("Loaded %d gold %s rows", len(df), split)

    if not sample or sample >= len(df):
        return df

    rng = np.random.RandomState(seed)

    must_keep_idx = set()
    for label in LABELS:
        pos_idx = df.index[df[label] == 1].tolist()
        n_keep = min(min_label_positives, len(pos_idx))
        if 0 < n_keep < len(pos_idx):
            chosen = rng.choice(pos_idx, size=n_keep, replace=False)
        else:
            chosen = pos_idx  # keep ALL positives if fewer than min_label_positives
        must_keep_idx.update(int(i) for i in chosen)
        logger.info(
            "  Label '%s': %d positives available in full split, keeping %d in sample",
            label, len(pos_idx), len(chosen),
        )

    must_keep_idx = sorted(must_keep_idx)
    n_must_keep = len(must_keep_idx)

    if n_must_keep >= sample:
        logger.warning(
            "Must-keep rows (%d) >= requested sample size (%d); "
            "truncating must-keep set randomly to fit.",
            n_must_keep, sample,
        )
        final_idx = sorted(rng.choice(must_keep_idx, size=sample, replace=False).tolist())
        df_sample = df.loc[final_idx].reset_index(drop=True)
        logger.info("Sampled %d rows (stratified, must-keep only)", len(df_sample))
        return df_sample

    remaining_budget = sample - n_must_keep
    remaining_pool = np.array(df.index.difference(must_keep_idx))
    fill_idx = rng.choice(remaining_pool, size=remaining_budget, replace=False)

    final_idx = sorted(set(must_keep_idx) | set(int(i) for i in fill_idx))
    df_sample = df.loc[final_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Sampled %d rows (stratified: %d must-keep + %d random fill)",
        len(df_sample), n_must_keep, remaining_budget,
    )

    return df_sample



def build_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    model_slug = args.model.replace("-", "_").replace(".", "")
    suffix = ""
    if args.cot:
        suffix += "_cot"
    if args.platform:
        suffix += f"_{args.platform}"
    if args.no_few_shot:
        suffix += "_nofs"

    return Path(f"results/llm_{model_slug}_{args.split}{suffix}")


def main():
    args = parse_args()

    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache_path or str(output_dir / "cache.jsonl")
    predictions_path = str(output_dir / "predictions")

    df = load_split(args.split, sample=args.sample, seed=args.seed)

    platform_rules = get_platform_rules(args.platform) if args.platform else None

    engine = LLMInferenceEngine(
        api_key=args.api_key,
        model_name=args.model,
        requests_per_minute=args.rpm,
        use_cot=args.cot,
        use_few_shot=not args.no_few_shot,
        include_reasoning=not args.no_reasoning,
        platform_rules=platform_rules,
        cache_path=cache_path,
    )

    # Cost estimate (always shown)
    cost_est = engine.estimate_cost(df)
    logger.info("Cost estimate:")
    for k, v in cost_est.items():
        logger.info("  %s: %s", k, v)

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # ── Run inference ──────────────────────────────────────────────────────────
    results = engine.predict_df(
        df=df,
        text_col="text",
        output_path=predictions_path,
        label_names=LABELS,
    )

    preds = results["predictions"]
    confs = results["confidences"]
    success = results["success_mask"]

    logger.info(
        "Inference done: %d/%d successful (%.1f%%)",
        success.sum(), len(success), 100 * success.mean(),
    )

    # ── Prepare y_true ─────────────────────────────────────────────────────────
    y_true = df[LABELS].values.astype(float)

    # ── Bootstrap CI report ────────────────────────────────────────────────────
    logger.info("Computing bootstrap CIs...")
    report = bootstrap_ci_report(
        y_true=y_true,
        y_pred=preds,
        n_resamples=1000,
        ci=0.95,
        random_state=42,
        label_names=LABELS,
    )

    print_ci_report(report, label_names=LABELS)

    # ── Save CI report as CSV ─────────────────────────────────────────────────
    ci_rows = []
    for label in LABELS:
        r = report[label]
        ci_rows.append({
            "label": label,
            "f1_mean": r["f1_mean"], "f1_lower": r["f1_lower"], "f1_upper": r["f1_upper"],
            "s_score_mean": r["s_score_mean"], "s_score_lower": r["s_score_lower"],
            "s_score_upper": r["s_score_upper"],
            "n_positives": r["n_positives"], "n_total": r["n_total"],
        })
    for key in ["macro_f1", "macro_s_score"]:
        r = report[key]
        metric = "f1" if key == "macro_f1" else "s_score"
        ci_rows.append({
            "label": key,
            f"{metric}_mean": r["mean"], f"{metric}_lower": r["lower"],
            f"{metric}_upper": r["upper"],
        })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(output_dir / "bootstrap_ci_report.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "split": args.split,
        "n_total": len(df),
        "n_successful": int(success.sum()),
        "n_failed": int((~success).sum()),
        "use_cot": args.cot,
        "use_few_shot": not args.no_few_shot,
        "platform": args.platform,
        "macro_f1": {
            "mean": report["macro_f1"]["mean"],
            "lower": report["macro_f1"]["lower"],
            "upper": report["macro_f1"]["upper"],
        },
        "macro_s_score": {
            "mean": report["macro_s_score"]["mean"],
            "lower": report["macro_s_score"]["lower"],
            "upper": report["macro_s_score"]["upper"],
        },
        "per_label": {
            label: {
                "f1": report[label]["f1_mean"],
                "s_score": report[label]["s_score_mean"],
                "n_positives": report[label]["n_positives"],
            }
            for label in LABELS
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    logger.info("Results saved to %s", output_dir)
    logger.info(
        "Macro S-Score: %.4f [%.4f, %.4f]",
        report["macro_s_score"]["mean"],
        report["macro_s_score"]["lower"],
        report["macro_s_score"]["upper"],
    )


if __name__ == "__main__":
    main()