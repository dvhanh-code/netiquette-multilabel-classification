"""
experiments/run_hybrid_eval.py
--------------------------------
Fuse BERT and LLM predictions, evaluate all fusion strategies, and
produce a full comparison table for the thesis (RQ3, RQ4).

Prerequisites:
  1. BERT experiment directory with:
       test_logits.npz   (required)
       thresholds.json   (required)
       val_logits.npz    (optional — needed for Weighted/Stacking fusion)

  2. LLM experiment directory with:
       predictions.npz   (required)

Usage:
    # Minimal (no val_logits needed):
    python3 experiments/run_hybrid_eval.py \\
        --bert-dir results/gbert_large_gold_silver_128_focal_lr5e6 \\
        --llm-dir  results/llm_gemini_flash_full \\
        --output-dir results/hybrid_e7_gemini

    # Full (with val_logits for Weighted + Stacking):
    python3 experiments/run_hybrid_eval.py \\
        --bert-dir results/gbert_large_gold_silver_128_focal_lr5e6 \\
        --llm-dir  results/llm_gemini_flash_full \\
        --output-dir results/hybrid_e7_gemini \\
        --use-weighted-stacking

Strategies:
    Always active:    Average, Confidence-gated, Union, Intersection
    Requires val set: Weighted, Stacking (enabled with --use-weighted-stacking)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.transformer_metrics import sigmoid
from src.hybrid.fusion import (
    AverageFusion, WeightedFusion, ConfidenceGatedFusion,
    StackingFusion, UnionFusion, IntersectionFusion,
)
from src.evaluation.compare import SystemComparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["hate_speech", "toxic", "threat", "insult"]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid BERT+LLM fusion evaluation")
    parser.add_argument("--bert-dir", required=True,
                        help="BERT experiment directory (test_logits.npz + thresholds.json)")
    parser.add_argument("--llm-dir", required=True,
                        help="LLM experiment directory (predictions.npz)")
    parser.add_argument("--output-dir", default="results/hybrid_comparison",
                        help="Output directory")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--n-resamples", type=int, default=1000,
                        help="Bootstrap resamples for confidence intervals")
    parser.add_argument("--use-weighted-stacking", action="store_true",
                        help="Enable Weighted + Stacking fusion (requires val_logits.npz)")
    return parser.parse_args()


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_bert_results(bert_dir: Path, split: str) -> Dict[str, np.ndarray]:
    npz_path = bert_dir / f"{split}_logits.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"BERT logits not found: {npz_path}\n"
            "Generate with: np.savez(path, logits=..., labels=..., label_mask=...)"
        )
    data = np.load(npz_path)
    logger.info("Loaded BERT %s logits: %s rows", split, data["logits"].shape[0])
    return {
        "logits":      data["logits"],
        "labels":      data["labels"],
        "label_mask":  data["label_mask"],
    }


def load_llm_results(llm_dir: Path) -> Dict[str, np.ndarray]:
    npz_path = llm_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"LLM predictions not found: {npz_path}")
    data = np.load(npz_path)
    logger.info("Loaded LLM predictions: %s rows", data["predictions"].shape[0])
    return {
        "predictions":  data["predictions"],
        "confidences":  data["confidences"],
        "success_mask": data["success_mask"],
    }


def load_bert_thresholds(bert_dir: Path) -> Dict[str, float]:
    thresh_path = bert_dir / "thresholds.json"
    if not thresh_path.exists():
        logger.warning("thresholds.json not found — using 0.5 for all labels")
        return {label: 0.5 for label in LABELS}
    with thresh_path.open() as f:
        thresholds = json.load(f)
    logger.info("Loaded thresholds: %s", thresholds)
    return thresholds


def apply_bert_thresholds(probs: np.ndarray,
                          thresholds: Dict[str, float]) -> np.ndarray:
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(LABELS):
        preds[:, i] = (probs[:, i] >= thresholds.get(label, 0.5)).astype(int)
    return preds


def load_val_data(bert_dir: Path,
                  llm_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load val set logits for fitting Weighted/Stacking fusion.
    Returns None if either file is missing (graceful degradation).
    """
    bert_val_path = bert_dir / "val_logits.npz"
    llm_val_path  = llm_dir  / "val_predictions.npz"

    if not bert_val_path.exists():
        logger.warning(
            "val_logits.npz not found in %s — "
            "Weighted/Stacking fusion unavailable.", bert_dir
        )
        return None

    if not llm_val_path.exists():
        # Try sibling directory named with 'val'
        for sibling in llm_dir.parent.iterdir():
            if sibling.is_dir() and "val" in sibling.name.lower():
                candidate = sibling / "predictions.npz"
                if candidate.exists():
                    llm_val_path = candidate
                    logger.info("Found LLM val predictions: %s", llm_val_path)
                    break
        else:
            logger.warning(
                "LLM val predictions not found — "
                "Weighted/Stacking fusion unavailable."
            )
            return None

    bert_val  = np.load(bert_val_path)
    llm_val   = np.load(llm_val_path)

    return {
        "bert_probs_val": sigmoid(bert_val["logits"]),
        "llm_confs_val":  llm_val["confidences"],
        "y_true_val":     bert_val["labels"].astype(float),
        "label_mask_val": bert_val["label_mask"].astype(bool),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    bert_dir   = Path(args.bert_dir)
    llm_dir    = Path(args.llm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading BERT results from %s", bert_dir)
    bert_data    = load_bert_results(bert_dir, args.split)
    bert_logits  = bert_data["logits"]
    y_true       = bert_data["labels"].astype(float)
    label_mask   = bert_data["label_mask"].astype(bool)
    bert_probs   = sigmoid(bert_logits)
    bert_thres   = load_bert_thresholds(bert_dir)
    bert_preds   = apply_bert_thresholds(bert_probs, bert_thres)

    logger.info("Loading LLM results from %s", llm_dir)
    llm_data     = load_llm_results(llm_dir)
    llm_preds    = llm_data["predictions"]
    llm_confs    = llm_data["confidences"]

    # ── Strategies ─────────────────────────────────────────────────────────────
    # These 4 always work — no val set needed
    strategies = {
        "Hybrid: Average":          AverageFusion(threshold=0.5),
        "Hybrid: Confidence-gated": ConfidenceGatedFusion(uncertainty_radius=0.2),
        "Hybrid: Union":            UnionFusion(),
        "Hybrid: Intersection":     IntersectionFusion(),
    }

    # Weighted + Stacking: only if val data available and flag set
    if args.use_weighted_stacking:
        val_data = load_val_data(bert_dir, llm_dir)
        if val_data is not None:
            weighted = WeightedFusion()
            stacking = StackingFusion()
            for strategy, name in [(weighted, "Hybrid: Weighted"),
                                   (stacking, "Hybrid: Stacking")]:
                logger.info("Fitting %s on val set...", name)
                strategy.fit(
                    bert_probs_val=val_data["bert_probs_val"],
                    llm_confs_val=val_data["llm_confs_val"],
                    y_true_val=val_data["y_true_val"],
                    label_mask_val=val_data["label_mask_val"],
                    label_names=LABELS,
                )
                strategies[name] = strategy
        else:
            logger.warning(
                "--use-weighted-stacking requested but val data unavailable. "
                "Skipping Weighted and Stacking strategies."
            )
    else:
        logger.info(
            "Weighted/Stacking fusion skipped "
            "(add --use-weighted-stacking to enable)."
        )

    # ── Comparison ─────────────────────────────────────────────────────────────
    cmp = SystemComparison(
        y_true=y_true,
        label_mask=label_mask,
        label_names=LABELS,
        n_resamples=args.n_resamples,
    )

    cmp.add_system("BERT (tuned thresholds)", bert_preds, bert_probs)
    cmp.add_system("LLM only (Gemini Flash)", llm_preds, llm_confs)

    for name, strategy in strategies.items():
        fused = strategy.fuse(bert_probs, llm_confs)
        cmp.add_system(name, fused.astype(float))
        logger.info("Evaluated: %s", name)

    # ── Results ────────────────────────────────────────────────────────────────
    table = cmp.comparison_table()
    cmp.print_table(table)

    # McNemar: BERT vs each hybrid
    for hybrid_name in strategies.keys():
        try:
            mcnemar_df = cmp.mcnemar_pairwise("BERT (tuned thresholds)", hybrid_name)
            if not mcnemar_df.empty:
                cmp.print_mcnemar(mcnemar_df)
        except Exception as e:
            logger.warning("McNemar test failed for %s: %s", hybrid_name, e)

    # Save comparison table
    table.to_csv(output_dir / "comparison_table.csv", index=False)
    logger.info("Saved comparison table → %s", output_dir / "comparison_table.csv")

    # Best system summary
    macro_rows = table[table["label"] == "MACRO"].copy()
    best_idx   = macro_rows["s_score"].idxmax()
    best_system = macro_rows.loc[best_idx, "system"]
    best_score  = macro_rows.loc[best_idx, "s_score"]

    logger.info("Best system: %s (Macro S=%.4f)", best_system, best_score)

    summary = {
        "split":               args.split,
        "bert_dir":            str(bert_dir),
        "llm_dir":             str(llm_dir),
        "bert_thresholds":     bert_thres,
        "strategies_used":     list(strategies.keys()),
        "best_system":         best_system,
        "best_macro_s_score":  float(best_score),
        "macro_by_system": macro_rows[[c for c in ["system", "s_score", "f1", "f2"] if c in macro_rows.columns]].to_dict("records"),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    logger.info("All results saved → %s", output_dir)


if __name__ == "__main__":
    main()