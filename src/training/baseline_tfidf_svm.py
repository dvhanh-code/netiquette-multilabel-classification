"""
src/training/baseline_tfidf_svm.py
----------------------------------
Classical baseline for multilabel Netiquette classification.

Model:
    TF-IDF + Linear SVM

Important:
    This dataset is partially labeled.
    NaN labels are not treated as 0.
    For each label, only annotated rows are used.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/final/unified_final_v1.parquet")
OUT_DIR = Path("results/baseline_tfidf_svm")
MODEL_DIR = OUT_DIR / "models"

LABELS = ["hate_speech", "toxic", "threat", "insult"]


def compute_metrics(y_true, y_pred):
    """Compute binary classification metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    mcc_norm = (mcc + 1) / 2
    s_score = (f2 + mcc_norm) / 2

    support_pos = int((y_true == 1).sum())
    support_neg = int((y_true == 0).sum())

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "mcc_norm": mcc_norm,
        "s_score": s_score,
        "support_pos": support_pos,
        "support_neg": support_neg,
        "support_total": int(len(y_true)),
    }


def build_model():
    """Build TF-IDF + Linear SVM pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                max_features=200_000,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=5000,
            ),
        ),
    ])


def evaluate_label(model, df, label, split_name):
    """Evaluate one label on rows where the label is annotated."""
    eval_df = df[df[label].notna()].copy()

    if len(eval_df) == 0:
        raise ValueError(f"No annotated rows for label {label} in {split_name}")

    x = eval_df["text"].fillna("").astype(str)
    y_true = eval_df[label].astype(int)

    y_pred = model.predict(x)

    metrics = compute_metrics(y_true, y_pred)
    metrics["label"] = label
    metrics["split"] = split_name

    return metrics


def macro_average(metrics_df):
    """Compute macro average over labels."""
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "mcc_norm",
        "s_score",
    ]

    row = {"label": "MACRO", "split": metrics_df["split"].iloc[0]}
    for col in metric_cols:
        row[col] = metrics_df[col].mean()

    row["support_pos"] = int(metrics_df["support_pos"].sum())
    row["support_neg"] = int(metrics_df["support_neg"].sum())
    row["support_total"] = int(metrics_df["support_total"].sum())

    return row


def train_one_label(train_df, label):
    """Train one binary classifier for one label."""
    label_train = train_df[train_df[label].notna()].copy()

    x_train = label_train["text"].fillna("").astype(str)
    y_train = label_train[label].astype(int)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    print(f"\nTraining label: {label}")
    print(f"  Train rows: {len(label_train):,}")
    print(f"  Positives : {pos:,}")
    print(f"  Negatives : {neg:,}")

    model = build_model()
    model.fit(x_train, y_train)

    return model


def print_metrics_table(title, df):
    """Print compact metric table."""
    print(f"\n{title}")
    print("-" * 90)
    cols = [
        "label",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "s_score",
        "support_pos",
        "support_total",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE: TF-IDF + Linear SVM")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nData quality:")
    print(df["data_quality"].value_counts())

    train_df = df[df["split"] == "train"].copy()
    val_df = df[(df["split"] == "val") & (df["is_gold"] == True)].copy()
    test_df = df[(df["split"] == "test") & (df["is_gold"] == True)].copy()

    print("\nTraining/Evaluation setup:")
    print(f"  Train rows: {len(train_df):,}")
    print(f"  Val rows  : {len(val_df):,}  (gold only)")
    print(f"  Test rows : {len(test_df):,}  (gold only)")

    models = {}
    val_metrics = []
    test_metrics = []

    for label in LABELS:
        model = train_one_label(train_df, label)
        models[label] = model

        model_path = MODEL_DIR / f"{label}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved model: {model_path}")

        val_metrics.append(evaluate_label(model, val_df, label, "val"))
        test_metrics.append(evaluate_label(model, test_df, label, "test"))

    val_df_metrics = pd.DataFrame(val_metrics)
    test_df_metrics = pd.DataFrame(test_metrics)

    val_macro = macro_average(val_df_metrics)
    test_macro = macro_average(test_df_metrics)

    val_df_metrics = pd.concat(
        [val_df_metrics, pd.DataFrame([val_macro])],
        ignore_index=True,
    )
    test_df_metrics = pd.concat(
        [test_df_metrics, pd.DataFrame([test_macro])],
        ignore_index=True,
    )

    val_path = OUT_DIR / "val_metrics.csv"
    test_path = OUT_DIR / "test_metrics.csv"

    val_df_metrics.to_csv(val_path, index=False)
    test_df_metrics.to_csv(test_path, index=False)

    print_metrics_table("VALIDATION METRICS", val_df_metrics)
    print_metrics_table("TEST METRICS", test_df_metrics)

    summary = {
        "model": "TF-IDF + LinearSVM",
        "dataset": str(DATA_PATH),
        "labels": LABELS,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_macro": {
            "precision": float(val_macro["precision"]),
            "recall": float(val_macro["recall"]),
            "f1": float(val_macro["f1"]),
            "f2": float(val_macro["f2"]),
            "mcc": float(val_macro["mcc"]),
            "s_score": float(val_macro["s_score"]),
        },
        "test_macro": {
            "precision": float(test_macro["precision"]),
            "recall": float(test_macro["recall"]),
            "f1": float(test_macro["f1"]),
            "f2": float(test_macro["f2"]),
            "mcc": float(test_macro["mcc"]),
            "s_score": float(test_macro["s_score"]),
        },
    }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {summary_path}")
    print(f"  {MODEL_DIR}/{{label}}.joblib")


if __name__ == "__main__":
    main()