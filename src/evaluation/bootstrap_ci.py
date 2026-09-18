"""
src/evaluation/bootstrap_ci.py
--------------------------------
Bootstrap confidence intervals for multilabel classification metrics.

Usage:
    from src.evaluation.bootstrap_ci import (
        bootstrap_ci_report,
        bootstrap_ci_single_label,
        print_ci_report,
        s_score,
    )
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

LABELS = ["hate_speech", "toxic", "threat", "insult"]

# Labels with fewer than this many positives in the test set are flagged
_UNSTABLE_THRESHOLD = 50


# ─────────────────────────────────────────────────────────────────────────────
# Core metric
# ─────────────────────────────────────────────────────────────────────────────

def s_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    S-Score from HSD-Detector paper (Ippen Digital).
    S = (F2_binary + M_norm) / 2
    where M_norm = (MCC + 1) / 2

    Uses binary (not macro) F2 and MCC.
    Returns 0.25 (floor value) if undefined
    (e.g. all-negative predictions or all-negative ground truth).

    Args:
        y_true: 1D binary integer array, shape (n,)
        y_pred: 1D binary integer array, shape (n,)
    """
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0.0)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    m_norm = (mcc + 1.0) / 2.0
    return float((f2 + m_norm) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-label bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci_single_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap CI for a single binary label.

    Args:
        y_true: 1D binary array (annotated rows only, no NaN)
        y_pred: 1D binary array of predictions
        n_resamples: number of bootstrap iterations
        ci: confidence level (default 0.95)
        random_state: for reproducibility

    Returns:
        dict with keys:
            {metric}_mean, {metric}_lower, {metric}_upper
            for metrics: f1, f2, precision, recall, s_score
            plus: n_positives, n_total
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    buckets: Dict[str, List[float]] = {
        "f1":        [],
        "f2":        [],
        "precision": [],
        "recall":    [],
        "s_score":   [],
    }

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        # Skip resamples with no positives — metric undefined
        if yt.sum() == 0:
            continue

        buckets["f1"].append(f1_score(yt, yp, zero_division=0.0))
        buckets["f2"].append(fbeta_score(yt, yp, beta=2, zero_division=0.0))
        buckets["precision"].append(precision_score(yt, yp, zero_division=0.0))
        buckets["recall"].append(recall_score(yt, yp, zero_division=0.0))
        buckets["s_score"].append(s_score(yt, yp))

    alpha = 1.0 - ci
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    out: Dict[str, float] = {
        "n_positives": int(y_true.sum()),
        "n_total":     int(n),
    }

    for metric, values in buckets.items():
        if not values:
            out[f"{metric}_mean"]  = float("nan")
            out[f"{metric}_lower"] = float("nan")
            out[f"{metric}_upper"] = float("nan")
        else:
            arr = np.array(values)
            out[f"{metric}_mean"]  = float(arr.mean())
            out[f"{metric}_lower"] = float(np.percentile(arr, lo_pct))
            out[f"{metric}_upper"] = float(np.percentile(arr, hi_pct))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full multilabel report
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_binary: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    label_names: List[str] = LABELS,
) -> Dict:
    """
    Full bootstrap CI report for multilabel classification.

    Args:
        y_true:        shape (n, num_labels), float with NaN for missing labels
        y_pred:        shape (n, num_labels), binary predictions {0, 1}
        y_true_binary: ignored (kept for API compatibility)
        n_resamples:   bootstrap iterations (default 1000)
        ci:            confidence level (default 0.95)
        random_state:  for reproducibility
        label_names:   list of label strings

    Returns:
        dict with per-label CI dicts + "macro_f1" and "macro_s_score" entries
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    report: Dict = {}

    # ── Per-label CIs ─────────────────────────────────────────────────────────
    for i, label in enumerate(label_names):
        yt_col = y_true[:, i]
        yp_col = y_pred[:, i]

        annotated = ~np.isnan(yt_col)
        yt_ann = yt_col[annotated].astype(int)
        yp_ann = yp_col[annotated].astype(int)

        report[label] = bootstrap_ci_single_label(
            yt_ann,
            yp_ann,
            n_resamples=n_resamples,
            ci=ci,
            random_state=random_state + i,
        )

    # ── Macro bootstrap (row-level resampling) ────────────────────────────────
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    n_labels = len(label_names)

    alpha = 1.0 - ci
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    macro_f1_boot:  List[float] = []
    macro_s_boot:   List[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]

        f1s: List[float] = []
        ss:  List[float] = []

        for i in range(n_labels):
            ann = ~np.isnan(yt_b[:, i])
            yt = yt_b[ann, i].astype(int)
            yp = yp_b[ann, i].astype(int)

            if yt.sum() == 0:
                continue

            f1s.append(f1_score(yt, yp, zero_division=0.0))
            ss.append(s_score(yt, yp))

        if f1s:
            macro_f1_boot.append(float(np.mean(f1s)))
            macro_s_boot.append(float(np.mean(ss)))

    def _ci_dict(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
        arr = np.array(values)
        return {
            "mean":  float(arr.mean()),
            "lower": float(np.percentile(arr, lo_pct)),
            "upper": float(np.percentile(arr, hi_pct)),
        }

    report["macro_f1"]      = _ci_dict(macro_f1_boot)
    report["macro_s_score"] = _ci_dict(macro_s_boot)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_ci_report(
    report: Dict,
    label_names: List[str] = LABELS,
) -> None:
    """Pretty print the CI report to console."""

    print()
    print("=" * 72)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%, n=1000 resamples)")
    print("=" * 72)
    print(
        f"  {'Label':<14} {'F1':>6} {'95% CI':>16} "
        f"{'S-Score':>8} {'95% CI':>16} "
        f"{'Pos':>5} {'N':>6}"
    )
    print("  " + "─" * 70)

    for label in label_names:
        r = report[label]
        f1_mean = r["f1_mean"]
        s_mean  = r["s_score_mean"]

        f1_str = f"{f1_mean:.3f}" if not np.isnan(f1_mean) else "  NaN"
        s_str  = f"{s_mean:.3f}"  if not np.isnan(s_mean)  else "  NaN"

        def _ci(lo, hi):
            if np.isnan(lo):
                return "[    NaN,     NaN]"
            return f"[{lo:.3f}, {hi:.3f}]"

        f1_ci = _ci(r["f1_lower"],      r["f1_upper"])
        s_ci  = _ci(r["s_score_lower"], r["s_score_upper"])

        flag = " ⚠" if r["n_positives"] < _UNSTABLE_THRESHOLD else ""
        print(
            f"  {label:<14} {f1_str:>6} {f1_ci:>16} "
            f"{s_str:>8} {s_ci:>16} "
            f"{r['n_positives']:>5} {r['n_total']:>6}{flag}"
        )

    print("  " + "─" * 70)

    mf1 = report["macro_f1"]
    ms  = report["macro_s_score"]

    mf1_str = f"{mf1['mean']:.3f}" if not np.isnan(mf1["mean"]) else "NaN"
    ms_str  = f"{ms['mean']:.3f}"  if not np.isnan(ms["mean"])  else "NaN"
    mf1_ci  = f"[{mf1['lower']:.3f}, {mf1['upper']:.3f}]"
    ms_ci   = f"[{ms['lower']:.3f}, {ms['upper']:.3f}]"

    print(
        f"  {'Macro':<14} {mf1_str:>6} {mf1_ci:>16} "
        f"{ms_str:>8} {ms_ci:>16}"
    )
    print("=" * 72)
    print("  ⚠  = statistically unstable (n_positives < 50)")
    print()
