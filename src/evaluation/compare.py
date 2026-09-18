"""
src/evaluation/compare.py
--------------------------
Multi-system comparison with bootstrap CIs and McNemar's test.

This module consolidates results from BERT, LLM, and hybrid systems
into a single comparison table, enabling statistically rigorous
reporting for RQ3 (hybrid vs. individual models).

Usage:
    from src.evaluation.compare import SystemComparison

    cmp = SystemComparison(y_true=test_labels, label_mask=test_mask)
    cmp.add_system("BERT gold_silver", bert_preds, bert_probs)
    cmp.add_system("LLM Gemini Flash", llm_preds, llm_confs)
    cmp.add_system("Hybrid Stacking",  hybrid_preds)

    table = cmp.comparison_table()
    cmp.print_table(table)
    cmp.save(table, "results/system_comparison.csv")

    pairwise = cmp.mcnemar_pairwise()
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.evaluation.bootstrap_ci import bootstrap_ci_report, print_ci_report

LABELS = ["hate_speech", "toxic", "threat", "insult"]


@dataclass
class SystemResult:
    name: str
    predictions: np.ndarray   # shape (n, k), float — 0 / 1 / NaN
    confidences: Optional[np.ndarray] = None  # shape (n, k), float
    metadata: Dict = field(default_factory=dict)


class SystemComparison:
    """
    Collects predictions from multiple systems and generates
    a comprehensive comparison table with CIs and significance tests.

    Args:
        y_true:      shape (n, k) float, NaN for unannotated labels
        label_mask:  shape (n, k) bool/int, 1=annotated (optional; inferred from NaN)
        label_names: label column names
        n_resamples: bootstrap resamples
        ci:          confidence level
    """

    def __init__(
        self,
        y_true: np.ndarray,
        label_mask: Optional[np.ndarray] = None,
        label_names: List[str] = LABELS,
        n_resamples: int = 1000,
        ci: float = 0.95,
    ):
        self.y_true = np.asarray(y_true, dtype=float)
        self.label_names = label_names
        self.n_resamples = n_resamples
        self.ci = ci

        if label_mask is not None:
            self.label_mask = np.asarray(label_mask, dtype=bool)
        else:
            self.label_mask = ~np.isnan(self.y_true)

        self._systems: List[SystemResult] = []

    def add_system(
        self,
        name: str,
        predictions: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        **metadata,
    ) -> "SystemComparison":
        """Register a system's predictions."""
        preds = np.asarray(predictions, dtype=float)
        if preds.shape != self.y_true.shape:
            raise ValueError(
                f"System {name!r}: predictions shape {preds.shape} "
                f"!= y_true shape {self.y_true.shape}"
            )
        self._systems.append(SystemResult(
            name=name,
            predictions=preds,
            confidences=np.asarray(confidences, dtype=float) if confidences is not None else None,
            metadata=metadata,
        ))
        return self

    def _ci_report_for(self, system: SystemResult) -> Dict:
        return bootstrap_ci_report(
            y_true=self.y_true,
            y_pred=system.predictions,
            n_resamples=self.n_resamples,
            ci=self.ci,
            random_state=42,
            label_names=self.label_names,
        )

    def comparison_table(self) -> pd.DataFrame:
        """
        Build a DataFrame with one row per (system, label/macro) and
        columns: f1_mean, f1_lower, f1_upper, s_score_mean, s_score_lower, s_score_upper.
        """
        rows = []
        for system in self._systems:
            report = self._ci_report_for(system)

            for label in self.label_names:
                r = report[label]
                rows.append({
                    "system":         system.name,
                    "label":          label,
                    "f1":             r["f1_mean"],
                    "f1_ci_lo":       r["f1_lower"],
                    "f1_ci_hi":       r["f1_upper"],
                    "s_score":        r["s_score_mean"],
                    "s_score_ci_lo":  r["s_score_lower"],
                    "s_score_ci_hi":  r["s_score_upper"],
                    "n_positives":    r["n_positives"],
                    "n_total":        r["n_total"],
                })

            mf1 = report["macro_f1"]
            ms  = report["macro_s_score"]
            rows.append({
                "system":         system.name,
                "label":          "MACRO",
                "f1":             mf1["mean"],
                "f1_ci_lo":       mf1["lower"],
                "f1_ci_hi":       mf1["upper"],
                "s_score":        ms["mean"],
                "s_score_ci_lo":  ms["lower"],
                "s_score_ci_hi":  ms["upper"],
                "n_positives":    None,
                "n_total":        None,
            })

        return pd.DataFrame(rows)

    def print_table(self, table: Optional[pd.DataFrame] = None) -> None:
        """Pretty-print the comparison table."""
        if table is None:
            table = self.comparison_table()

        print()
        print("=" * 90)
        print("SYSTEM COMPARISON — S-Score and F1 (95% CI, n=1000 bootstrap resamples)")
        print("=" * 90)

        for label in self.label_names + ["MACRO"]:
            sub = table[table["label"] == label]
            print(f"\n  ── {label} ──")
            print(f"  {'System':<30} {'F1':>6} {'95% CI F1':>18} {'S-Score':>8} {'95% CI S':>18}")
            print("  " + "─" * 82)
            for _, row in sub.iterrows():
                f1_ci = f"[{row['f1_ci_lo']:.3f}, {row['f1_ci_hi']:.3f}]"
                s_ci  = f"[{row['s_score_ci_lo']:.3f}, {row['s_score_ci_hi']:.3f}]"
                print(
                    f"  {row['system']:<30} {row['f1']:>6.3f} {f1_ci:>18} "
                    f"{row['s_score']:>8.3f} {s_ci:>18}"
                )

        print()
        print("=" * 90)

    def save(
        self,
        table: Optional[pd.DataFrame] = None,
        path: str = "results/system_comparison.csv",
    ) -> None:
        if table is None:
            table = self.comparison_table()
        table.to_csv(path, index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Statistical significance: McNemar's test
    # ──────────────────────────────────────────────────────────────────────────

    def mcnemar_pairwise(
        self,
        system_a_name: Optional[str] = None,
        system_b_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        McNemar's test comparing all pairs (or a specific pair) of systems
        per label.

        McNemar tests whether two classifiers make significantly different errors
        on the same test examples. It uses only the discordant pairs:
          b = A correct, B wrong
          c = A wrong, B correct
          chi2 = (|b - c| - 1)^2 / (b + c), df=1

        Returns a DataFrame with columns:
            system_a, system_b, label, b, c, chi2, p_value, significant
        """
        if system_a_name and system_b_name:
            pairs = [(system_a_name, system_b_name)]
        else:
            names = [s.name for s in self._systems]
            pairs = [
                (names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
            ]

        rows = []
        for name_a, name_b in pairs:
            sys_a = self._get_system(name_a)
            sys_b = self._get_system(name_b)

            for li, label in enumerate(self.label_names):
                mask = self.label_mask[:, li]
                if mask.sum() == 0:
                    continue

                yt = self.y_true[mask, li].astype(int)
                ya = sys_a.predictions[mask, li]
                yb = sys_b.predictions[mask, li]

                # Skip NaN rows (LLM failures)
                valid = ~np.isnan(ya) & ~np.isnan(yb)
                if valid.sum() < 10:
                    continue

                yt = yt[valid]
                ya = ya[valid].astype(int)
                yb = yb[valid].astype(int)

                a_correct = (ya == yt)
                b_correct = (yb == yt)

                # Discordant pairs
                b_count = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
                c_count = int(np.sum(~a_correct & b_correct))   # A wrong, B right

                if b_count + c_count == 0:
                    chi2_stat, p_val = 0.0, 1.0
                else:
                    chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
                    p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

                rows.append({
                    "system_a":   name_a,
                    "system_b":   name_b,
                    "label":      label,
                    "n":          int(valid.sum()),
                    "b":          b_count,
                    "c":          c_count,
                    "chi2":       round(chi2_stat, 4),
                    "p_value":    round(p_val, 4),
                    "significant": p_val < 0.05,
                })

        return pd.DataFrame(rows)

    def _get_system(self, name: str) -> SystemResult:
        for s in self._systems:
            if s.name == name:
                return s
        available = [s.name for s in self._systems]
        raise ValueError(f"System {name!r} not found. Available: {available}")

    def print_mcnemar(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            df = self.mcnemar_pairwise()

        print()
        print("=" * 80)
        print("McNEMAR'S TEST (pairwise, per label, α=0.05)")
        print("=" * 80)

        for _, row in df.iterrows():
            sig = " *" if row["significant"] else ""
            print(
                f"  {row['system_a']!r:30s} vs {row['system_b']!r:30s} | "
                f"{row['label']:<12} | "
                f"b={row['b']:>4} c={row['c']:>4} | "
                f"χ²={row['chi2']:>6.3f} p={row['p_value']:.4f}{sig}"
            )
        print()
        print("  * significant at p<0.05")
        print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: CI-overlap significance check (non-parametric)
# ─────────────────────────────────────────────────────────────────────────────

def ci_overlap(
    lower_a: float,
    upper_a: float,
    lower_b: float,
    upper_b: float,
) -> Tuple[bool, float]:
    """
    Check if two CIs overlap and compute the overlap magnitude.

    No overlap is a conservative proxy for significance.
    McNemar is more statistically rigorous; use this for quick visual checks.

    Returns:
        (overlaps: bool, overlap_magnitude: float)
        overlap_magnitude < 0 means a gap of that size exists between the CIs.
    """
    overlap = min(upper_a, upper_b) - max(lower_a, lower_b)
    return overlap > 0, overlap