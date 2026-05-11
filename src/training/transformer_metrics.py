"""
src/training/transformer_metrics.py
-----------------------------------
Evaluation metrics for multilabel Transformer models.

Supports partially labeled multilabel data:
    - Metrics are computed only where label_mask == 1
    - Missing labels are ignored, not treated as negative

Metrics:
    precision, recall, F1, F2, MCC, normalized MCC, S-Score
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
)

LABELS = ["hate_speech", "toxic", "threat", "insult"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics for one binary label."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        mcc = 0.0
    else:
        mcc = matthews_corrcoef(y_true, y_pred)

    mcc_norm = (mcc + 1.0) / 2.0
    s_score = (f2 + mcc_norm) / 2.0

    support_pos = int((y_true == 1).sum())
    support_neg = int((y_true == 0).sum())

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "mcc": float(mcc),
        "mcc_norm": float(mcc_norm),
        "s_score": float(s_score),
        "support_pos": support_pos,
        "support_neg": support_neg,
        "support_total": int(len(y_true)),
    }


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    label_names: Optional[List[str]] = None,
    split_name: str = "eval",
) -> pd.DataFrame:
    """
    Compute per-label and macro metrics for multilabel predictions.

    Args:
        logits:
            Array with shape (n_samples, n_labels).
        labels:
            Array with shape (n_samples, n_labels).
            NaN labels should already be replaced by 0.
        label_mask:
            Array with shape (n_samples, n_labels).
            1 = annotated label, 0 = missing label.
        thresholds:
            Optional dict mapping label -> threshold.
            If None, threshold 0.5 is used for all labels.
        label_names:
            List of labels. Defaults to LABELS.
        split_name:
            Name of split, e.g. "val" or "test".

    Returns:
        DataFrame with one row per label plus one MACRO row.
    """
    label_names = label_names if label_names is not None else LABELS

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    label_mask = np.asarray(label_mask)

    if logits.shape != labels.shape:
        raise ValueError(f"logits shape {logits.shape} != labels shape {labels.shape}")

    if labels.shape != label_mask.shape:
        raise ValueError(f"labels shape {labels.shape} != label_mask shape {label_mask.shape}")

    if logits.shape[1] != len(label_names):
        raise ValueError(
            f"Number of labels in logits ({logits.shape[1]}) "
            f"does not match label_names ({len(label_names)})"
        )

    probs = sigmoid(logits)

    rows = []

    for i, label in enumerate(label_names):
        mask = label_mask[:, i].astype(bool)

        if mask.sum() == 0:
            row = {
                "label": label,
                "split": split_name,
                "threshold": thresholds.get(label, 0.5) if thresholds else 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f2": 0.0,
                "mcc": 0.0,
                "mcc_norm": 0.5,
                "s_score": 0.25,
                "support_pos": 0,
                "support_neg": 0,
                "support_total": 0,
            }
            rows.append(row)
            continue

        threshold = thresholds.get(label, 0.5) if thresholds else 0.5

        y_true = labels[mask, i].astype(int)
        y_pred = (probs[mask, i] >= threshold).astype(int)

        metrics = compute_binary_metrics(y_true, y_pred)
        metrics["label"] = label
        metrics["split"] = split_name
        metrics["threshold"] = float(threshold)

        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    macro_cols = [
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "mcc_norm",
        "s_score",
    ]

    macro_row = {
        "label": "MACRO",
        "split": split_name,
        "threshold": np.nan,
    }

    for col in macro_cols:
        macro_row[col] = float(metrics_df[col].mean())

    macro_row["support_pos"] = int(metrics_df["support_pos"].sum())
    macro_row["support_neg"] = int(metrics_df["support_neg"].sum())
    macro_row["support_total"] = int(metrics_df["support_total"].sum())

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([macro_row])],
        ignore_index=True,
    )

    ordered_cols = [
        "label",
        "split",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "mcc_norm",
        "s_score",
        "support_pos",
        "support_neg",
        "support_total",
    ]

    return metrics_df[ordered_cols]


def tune_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    label_names: Optional[List[str]] = None,
    metric: str = "f2",
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Tune one threshold per label on validation data.

    Args:
        logits:
            Model logits, shape (n_samples, n_labels).
        labels:
            Ground-truth labels, shape (n_samples, n_labels).
        label_mask:
            Mask, shape (n_samples, n_labels).
        label_names:
            Label names.
        metric:
            Metric to optimize: "f1", "f2", or "s_score".
        thresholds:
            Candidate thresholds. Defaults to 0.05 ... 0.95.

    Returns:
        Dict label -> best threshold.
    """
    label_names = label_names if label_names is not None else LABELS

    if metric not in {"f1", "f2", "s_score"}:
        raise ValueError("metric must be one of: f1, f2, s_score")

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    probs = sigmoid(np.asarray(logits))
    labels = np.asarray(labels)
    label_mask = np.asarray(label_mask)

    best_thresholds = {}

    for i, label in enumerate(label_names):
        mask = label_mask[:, i].astype(bool)

        if mask.sum() == 0:
            best_thresholds[label] = 0.5
            continue

        y_true = labels[mask, i].astype(int)

        best_score = -1.0
        best_t = 0.5

        for t in thresholds:
            y_pred = (probs[mask, i] >= t).astype(int)
            metrics = compute_binary_metrics(y_true, y_pred)
            score = metrics[metric]

            if score > best_score:
                best_score = score
                best_t = float(t)

        best_thresholds[label] = best_t

    return best_thresholds


def print_metrics_table(title: str, metrics_df: pd.DataFrame) -> None:
    """Print compact metrics table."""
    print(f"\n{title}")
    print("-" * 100)

    cols = [
        "label",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "s_score",
        "support_pos",
        "support_total",
    ]

    print(
        metrics_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    # Small sanity test
    logits = np.array([
        [2.0, -1.0, 0.0, 1.0],
        [-1.0, 2.0, 0.0, -2.0],
        [0.5, 0.5, 2.0, -1.0],
    ])

    labels = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])

    label_mask = np.array([
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ])

    df = compute_multilabel_metrics(
        logits=logits,
        labels=labels,
        label_mask=label_mask,
        split_name="debug",
    )

    print_metrics_table("DEBUG METRICS", df)

    best = tune_thresholds(
        logits=logits,
        labels=labels,
        label_mask=label_mask,
        metric="f2",
    )

    print("\nBest thresholds:")
    print(best)
    print("\nOK")