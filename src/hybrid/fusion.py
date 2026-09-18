"""
src/hybrid/fusion.py
--------------------
Hybrid fusion strategies for combining BERT and LLM predictions.

BERT outputs sigmoid probabilities in [0, 1] per label.
LLM outputs binary predictions and confidences in [0, 1] per label.

Fusion strategies implemented:
  AverageFusion           simple average of probabilities, shared threshold
  WeightedFusion          per-label alpha * bert + (1-alpha) * llm
  ConfidenceGatedFusion   trust BERT when certain; fall back to LLM when uncertain
  StackingFusion          logistic meta-learner trained on val set (most principled)
  UnionFusion             positive if EITHER system predicts positive
  IntersectionFusion      positive only if BOTH systems predict positive

All strategies expose a unified interface:
    fuse(bert_probs, llm_confidences) -> np.ndarray  # shape (n, k), binary

Strategies that require fitting (StackingFusion, WeightedFusion with auto-alpha)
expose a fit(bert_probs_val, llm_confs_val, y_true_val, label_mask_val) method.

Usage:
    from src.hybrid.fusion import StackingFusion, WeightedFusion

    # Fit on val set
    fusion = StackingFusion()
    fusion.fit(bert_probs_val, llm_confs_val, y_true_val, label_mask_val)

    # Apply to test set
    preds = fusion.fuse(bert_probs_test, llm_confs_test)
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LABELS = ["hate_speech", "toxic", "threat", "insult"]


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class FusionStrategy(ABC):
    """
    Base class for all hybrid fusion strategies.

    All subclasses must implement fuse(). Strategies that need to fit
    parameters on validation data should also implement fit().
    """

    name: str = "base"

    @abstractmethod
    def fuse(
        self,
        bert_probs: np.ndarray,
        llm_confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse BERT and LLM outputs into binary predictions.

        Args:
            bert_probs:      shape (n, k), BERT sigmoid probabilities
            llm_confidences: shape (n, k), LLM confidences (NaN = failed call)

        Returns:
            shape (n, k), binary predictions {0, 1}
        """

    def fit(
        self,
        bert_probs_val: np.ndarray,
        llm_confs_val: np.ndarray,
        y_true_val: np.ndarray,
        label_mask_val: Optional[np.ndarray] = None,
        label_names: List[str] = LABELS,
    ) -> "FusionStrategy":
        """Override in subclasses that learn parameters from val data."""
        return self

    def _safe_merge(
        self,
        bert_probs: np.ndarray,
        llm_confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Where LLM confidence is NaN (failed call), fall back to BERT probability.
        Returns a merged confidence array with no NaN.
        """
        merged = llm_confidences.copy()
        nan_mask = np.isnan(merged)
        merged[nan_mask] = bert_probs[nan_mask]
        return merged

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Simple average
# ─────────────────────────────────────────────────────────────────────────────

class AverageFusion(FusionStrategy):
    """
    Fused score = (bert_prob + llm_confidence) / 2.
    Binary prediction by comparing to threshold.

    Simplest approach. No parameters to fit.
    Works well when BERT and LLM are well-calibrated.
    """

    name = "average"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        merged_llm = self._safe_merge(bert_probs, llm_confidences)
        fused = (bert_probs + merged_llm) / 2.0
        return (fused >= self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Weighted fusion (per-label alpha)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedFusion(FusionStrategy):
    """
    Fused score = alpha * bert_prob + (1 - alpha) * llm_confidence.

    alpha can be:
      - a scalar (same weight for all labels)
      - a dict {label: alpha} for per-label weights
      - None → fit from val data by grid search over [0, 1] in steps of 0.05
    """

    name = "weighted"

    def __init__(
        self,
        alpha: Optional[float] = None,
        threshold: float = 0.5,
        label_names: List[str] = LABELS,
    ):
        self.threshold = threshold
        self.label_names = label_names
        if alpha is not None:
            self.alphas: np.ndarray = np.full(len(label_names), alpha)
        else:
            self.alphas = np.full(len(label_names), 0.5)  # default; overridden by fit

        self._fitted = alpha is not None

    def fit(
        self,
        bert_probs_val: np.ndarray,
        llm_confs_val: np.ndarray,
        y_true_val: np.ndarray,
        label_mask_val: Optional[np.ndarray] = None,
        label_names: List[str] = LABELS,
    ) -> "WeightedFusion":
        """Grid-search alpha per label on val set, optimizing S-Score."""
        from src.evaluation.bootstrap_ci import s_score

        best_alphas = []
        candidate_alphas = np.arange(0.0, 1.01, 0.05)

        for i, label in enumerate(label_names):
            mask = label_mask_val[:, i].astype(bool) if label_mask_val is not None else \
                   ~np.isnan(y_true_val[:, i])
            if mask.sum() == 0:
                best_alphas.append(0.5)
                continue

            yt = y_true_val[mask, i].astype(int)
            bp = bert_probs_val[mask, i]
            lc = llm_confs_val[mask, i]

            # Replace NaN LLM with BERT fallback per-label
            nan_in_label = np.isnan(lc)
            lc = lc.copy()
            lc[nan_in_label] = bp[nan_in_label]

            best_score = -1.0
            best_a = 0.5
            for a in candidate_alphas:
                fused = a * bp + (1 - a) * lc
                yp = (fused >= self.threshold).astype(int)
                score = s_score(yt, yp)
                if score > best_score:
                    best_score = score
                    best_a = float(a)
            best_alphas.append(best_a)

        self.alphas = np.array(best_alphas)
        self._fitted = True
        return self

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        merged_llm = self._safe_merge(bert_probs, llm_confidences)
        fused = self.alphas * bert_probs + (1 - self.alphas) * merged_llm
        return (fused >= self.threshold).astype(int)

    def alpha_dict(self) -> Dict[str, float]:
        return dict(zip(self.label_names, self.alphas.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Confidence-gated fusion
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceGatedFusion(FusionStrategy):
    """
    Use BERT when it is confident; use LLM when BERT is uncertain.

    "Uncertainty" is defined as |bert_prob - 0.5| < uncertainty_radius.
    When BERT is uncertain: use LLM confidence with llm_threshold.
    When BERT is certain:   use BERT probability with bert_threshold.

    Rationale: BERT is fast and reliable for clear-cut cases. LLMs are better
    at nuanced cases where BERT hovers near the decision boundary.
    """

    name = "confidence_gated"

    def __init__(
        self,
        uncertainty_radius: float = 0.2,
        bert_threshold: float = 0.5,
        llm_threshold: float = 0.5,
    ):
        self.uncertainty_radius = uncertainty_radius
        self.bert_threshold = bert_threshold
        self.llm_threshold = llm_threshold

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        merged_llm = self._safe_merge(bert_probs, llm_confidences)

        bert_uncertain = np.abs(bert_probs - 0.5) < self.uncertainty_radius

        # Default: BERT decision
        result = (bert_probs >= self.bert_threshold).astype(int)

        # Override with LLM where BERT is uncertain
        llm_decision = (merged_llm >= self.llm_threshold).astype(int)
        result[bert_uncertain] = llm_decision[bert_uncertain]

        return result

    def uncertainty_rate(self, bert_probs: np.ndarray) -> float:
        """Fraction of predictions where LLM override is triggered."""
        return float(np.mean(np.abs(bert_probs - 0.5) < self.uncertainty_radius))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Stacking (meta-learner)
# ─────────────────────────────────────────────────────────────────────────────

class StackingFusion(FusionStrategy):
    """
    Train one logistic regression per label on val set features:
        [bert_prob, llm_confidence]

    This is the most principled hybrid approach because it:
      - Learns the optimal combination from labeled data
      - Handles miscalibration between BERT and LLM
      - Can be extended with additional features (text length, etc.)

    Requires calling .fit() on validation data before .fuse().
    """

    name = "stacking"

    def __init__(
        self,
        C: float = 1.0,
        label_names: List[str] = LABELS,
    ):
        self.C = C
        self.label_names = label_names
        self._classifiers: List[Optional[LogisticRegression]] = [None] * len(label_names)
        self._scalers: List[Optional[StandardScaler]] = [None] * len(label_names)
        self._fitted = False

    def fit(
        self,
        bert_probs_val: np.ndarray,
        llm_confs_val: np.ndarray,
        y_true_val: np.ndarray,
        label_mask_val: Optional[np.ndarray] = None,
        label_names: List[str] = LABELS,
    ) -> "StackingFusion":
        """Train per-label logistic regression on validation set."""
        for i, label in enumerate(label_names):
            mask = label_mask_val[:, i].astype(bool) if label_mask_val is not None else \
                   ~np.isnan(y_true_val[:, i])
            if mask.sum() < 10:
                warnings.warn(
                    f"Label {label!r} has only {mask.sum()} annotated val rows. "
                    "Stacking may be unreliable; falling back to average."
                )
                continue

            yt = y_true_val[mask, i].astype(int)
            bp = bert_probs_val[mask, i].reshape(-1, 1)
            lc = llm_confs_val[mask, i].reshape(-1, 1)

            # Replace NaN LLM with BERT fallback
            nan_mask = np.isnan(lc)
            lc = lc.copy()
            lc[nan_mask] = bp[nan_mask]

            X = np.hstack([bp, lc])

            if len(np.unique(yt)) < 2:
                warnings.warn(
                    f"Label {label!r} has only one class in val set. Skipping stacking."
                )
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(C=self.C, max_iter=500, random_state=42)
            clf.fit(X_scaled, yt)

            self._classifiers[i] = clf
            self._scalers[i] = scaler

        self._fitted = True
        return self

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .fuse()")

        n, k = bert_probs.shape
        result = np.zeros((n, k), dtype=int)

        for i in range(k):
            clf = self._classifiers[i]
            scaler = self._scalers[i]

            bp = bert_probs[:, i].reshape(-1, 1)
            lc = llm_confidences[:, i].reshape(-1, 1).copy()
            nan_mask = np.isnan(lc)
            lc[nan_mask] = bp[nan_mask]

            X = np.hstack([bp, lc])

            if clf is None or scaler is None:
                # Fallback to average if stacking failed for this label
                fused = (bp.flatten() + lc.flatten()) / 2.0
                result[:, i] = (fused >= 0.5).astype(int)
            else:
                X_scaled = scaler.transform(X)
                result[:, i] = clf.predict(X_scaled)

        return result

    def feature_weights(self) -> Dict[str, Dict[str, float]]:
        """Return logistic regression coefficients per label (BERT weight, LLM weight)."""
        out = {}
        for i, label in enumerate(self.label_names):
            clf = self._classifiers[i]
            if clf is not None:
                coefs = clf.coef_[0]
                out[label] = {"bert_weight": coefs[0], "llm_weight": coefs[1]}
            else:
                out[label] = {"bert_weight": None, "llm_weight": None}
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 & 6: Voting strategies
# ─────────────────────────────────────────────────────────────────────────────

class UnionFusion(FusionStrategy):
    """
    Positive if EITHER BERT or LLM predicts positive.
    Maximizes recall; minimizes false negatives.
    Useful for safety-critical moderation where missing positives is costly.
    """

    name = "union"

    def __init__(self, bert_threshold: float = 0.5, llm_threshold: float = 0.5):
        self.bert_threshold = bert_threshold
        self.llm_threshold = llm_threshold

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        merged_llm = self._safe_merge(bert_probs, llm_confidences)
        bert_preds = (bert_probs >= self.bert_threshold).astype(int)
        llm_preds = (merged_llm >= self.llm_threshold).astype(int)
        return np.maximum(bert_preds, llm_preds)


class IntersectionFusion(FusionStrategy):
    """
    Positive only if BOTH BERT and LLM predict positive.
    Maximizes precision; minimizes false positives.
    Useful when false positives (wrongly flagging clean content) are costly.
    """

    name = "intersection"

    def __init__(self, bert_threshold: float = 0.5, llm_threshold: float = 0.5):
        self.bert_threshold = bert_threshold
        self.llm_threshold = llm_threshold

    def fuse(self, bert_probs: np.ndarray, llm_confidences: np.ndarray) -> np.ndarray:
        merged_llm = self._safe_merge(bert_probs, llm_confidences)
        bert_preds = (bert_probs >= self.bert_threshold).astype(int)
        llm_preds = (merged_llm >= self.llm_threshold).astype(int)
        return np.minimum(bert_preds, llm_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all strategies for grid comparison
# ─────────────────────────────────────────────────────────────────────────────

def all_strategies(
    bert_threshold: float = 0.5,
    uncertainty_radius: float = 0.2,
) -> List[FusionStrategy]:
    """Return one instance of each strategy for comparison experiments."""
    return [
        AverageFusion(threshold=bert_threshold),
        WeightedFusion(threshold=bert_threshold),           # alpha=0.5 default; fit on val
        ConfidenceGatedFusion(uncertainty_radius=uncertainty_radius),
        StackingFusion(),                                    # must be fit on val
        UnionFusion(bert_threshold=bert_threshold),
        IntersectionFusion(bert_threshold=bert_threshold),
    ]