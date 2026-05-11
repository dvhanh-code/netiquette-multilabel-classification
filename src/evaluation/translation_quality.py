"""
src/evaluation/translation_quality.py
--------------------------------------
Evaluate translation quality of silver data using LaBSE
(Language-agnostic BERT Sentence Embeddings, Feng et al. 2020).

Input:
    data/processed/unified_with_quality_translated_with_metadata.parquet
    (produced by: python3 run_pipeline.py --translate --keep-translation-metadata)

Output:
    data/evaluation/translation_quality_LaBSE.xlsx  (3 sheets)
    Console report in German

Usage
-----
    python3 -c "from src.evaluation.translation_quality import main; main()"
    python3 -c "from src.evaluation.translation_quality import main; main(sample_n=200)"
"""

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

_REQUIRED_COLUMNS = {
    "text",
    "text_original",
    "translation_status",
    "data_quality",
    "is_gold",
    "source",
    "hate_speech",
    "toxic",
    "threat",
    "insult",
}

_LABELS = ["hate_speech", "toxic", "threat", "insult"]

# Quality category thresholds
_THRESH_GUT = 0.75
_THRESH_AKZEPTABEL = 0.55

# Excel output column order
_EXCEL_COLUMNS = [
    "text_original",
    "text",
    "labse_score",
    "qualitaet",
    "source",
    "hate_speech",
    "toxic",
    "threat",
    "insult",
]

# openpyxl rejects control characters in the range U+0000–U+001F except tab/LF/CR
_ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

# Rows above this threshold trigger parquet+summary path instead of full Excel
_LARGE_DATASET_THRESHOLD = 50_000


def _clean_excel_string(value):
    if isinstance(value, str):
        return _ILLEGAL_EXCEL_CHARS_RE.sub("", value)
    return value


def _sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(_clean_excel_string)
    return out


# ---------------------------------------------------------------------------
# Function 1: load_data
# ---------------------------------------------------------------------------

def load_data(metadata_path: Path) -> pd.DataFrame:
    """
    Load and filter the translation metadata parquet.

    Keeps only rows where:
        translation_status == "success"
        data_quality == "silver"
        text_original is not NaN and len >= 5
        text is not NaN and len >= 5

    Returns filtered DataFrame ready for LaBSE evaluation.
    """
    df = pd.read_parquet(metadata_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata file is missing required columns: {sorted(missing)}\n"
            f"Found: {list(df.columns)}"
        )

    # Keep only silver rows with successful translations
    df = df[
        (df["translation_status"] == "success") &
        (df["data_quality"] == "silver")
    ].copy()

    # Drop rows where either text field is missing or too short
    df = df.dropna(subset=["text_original", "text"])
    df["text_original"] = df["text_original"].astype(str)
    df["text"] = df["text"].astype(str)
    df = df[df["text_original"].str.strip().str.len() >= 5]
    df = df[df["text"].str.strip().str.len() >= 5]

    df = df.reset_index(drop=True)

    print(f"Verfügbare Zeilen für LaBSE-Evaluation: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Function 2: sample_for_evaluation
# ---------------------------------------------------------------------------

def sample_for_evaluation(
    df: pd.DataFrame,
    n: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Take a proportional sample stratified by source corpus.

    If total rows < n, all rows are used.
    Each source gets a share proportional to its size in df.
    """
    total = len(df)

    if total <= n:
        print(f"Gesamtzahl ({total:,}) ≤ n={n} — alle Zeilen werden verwendet.")
        print(f"Sample-Größe pro Quelle:")
        for source, count in df["source"].value_counts().items():
            print(f"  {source}: {count:,}")
        return df.copy()

    source_counts = df["source"].value_counts()
    fractions = source_counts / total

    parts = []
    for source, frac in fractions.items():
        k = max(1, round(frac * n))
        sub = df[df["source"] == source]
        k = min(k, len(sub))
        parts.append(sub.sample(n=k, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)

    # If rounding pushed us slightly over n, trim to n
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)

    sampled = sampled.reset_index(drop=True)

    print(f"Sample-Größe pro Quelle (gesamt {len(sampled):,} von {total:,}):")
    for source, count in sampled["source"].value_counts().items():
        print(f"  {source}: {count:,}")

    return sampled


# ---------------------------------------------------------------------------
# Function 3: compute_labse_scores
# ---------------------------------------------------------------------------

def compute_labse_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode text_original (EN) and text (DE) with LaBSE and compute
    row-wise cosine similarity.

    Adds column 'labse_score' (float, range [-1, 1], typically [0.5, 1.0]).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading LaBSE model...")
    model = SentenceTransformer(LABSE_MODEL_NAME)

    df = df.copy()

    print("Encoding English originals (text_original)...")
    embeddings_en = model.encode(
        df["text_original"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Encoding German translations (text)...")
    embeddings_de = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Normalized embeddings → dot product == cosine similarity
    scores = (embeddings_en * embeddings_de).sum(axis=1)
    df["labse_score"] = scores.astype(float)

    return df


# ---------------------------------------------------------------------------
# Function 4: categorize_scores
# ---------------------------------------------------------------------------

def categorize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a quality category to each row based on LaBSE score.

        score >= 0.75  -> "gut"
        score >= 0.55  -> "akzeptabel"
        score <  0.55  -> "schlecht"

    Adds column 'qualitaet'.
    """
    df = df.copy()

    conditions = [
        df["labse_score"] >= _THRESH_GUT,
        df["labse_score"] >= _THRESH_AKZEPTABEL,
    ]
    choices = ["gut", "akzeptabel"]
    df["qualitaet"] = np.select(conditions, choices, default="schlecht")

    return df


# ---------------------------------------------------------------------------
# Function 4b: filter_by_label
# ---------------------------------------------------------------------------

def filter_by_label(
    df: pd.DataFrame,
    label: Optional[str] = None,
    positive_only: bool = False,
) -> pd.DataFrame:
    """
    Optionally restrict the DataFrame to rows relevant for a specific label.

    Args:
        label:         One of _LABELS, or None (returns df unchanged).
        positive_only: If True, keep only rows where label == 1.0.
                       If False, keep rows where label is not NaN.

    Returns a filtered copy.
    """
    if label is None:
        return df

    if label not in _LABELS:
        raise ValueError(f"Unknown label '{label}'. Must be one of {_LABELS}.")

    if positive_only:
        filtered = df[df[label] == 1.0].copy()
        mode = "Positiv-Beispiele"
    else:
        filtered = df[df[label].notna()].copy()
        mode = "annotierte Zeilen"

    filtered = filtered.reset_index(drop=True)
    print(f"Label-Filter '{label}' ({mode}): {len(filtered):,} Zeilen verbleiben.")
    return filtered


# ---------------------------------------------------------------------------
# Function 5: print_report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, total_silver_rows: Optional[int] = None) -> None:
    """
    Print a German-language evaluation report to stdout.

    Args:
        df:                 Scored and categorized sample DataFrame.
        total_silver_rows:  Total silver rows in the full dataset (for extrapolation).
                            If None, uses len(df).
    """
    n = len(df)
    if total_silver_rows is None:
        total_silver_rows = n

    scores = df["labse_score"]
    quality_counts = df["qualitaet"].value_counts()

    n_gut        = int(quality_counts.get("gut",        0))
    n_akzeptabel = int(quality_counts.get("akzeptabel", 0))
    n_schlecht   = int(quality_counts.get("schlecht",   0))

    pct = lambda k: f"{100.0 * k / n:.1f}" if n > 0 else "0.0"

    bad_rate = n_schlecht / n if n > 0 else 0.0
    expected_bad = round(bad_rate * total_silver_rows)

    sep = "═" * 52

    print(f"\n{sep}")
    print("ÜBERSETZUNGSQUALITÄT — LaBSE Evaluationsbericht")
    print(f"Modell: {LABSE_MODEL_NAME} (Feng et al., 2020)")
    print(sep)

    print(f"\nAusgewertete Zeilen: {n:,}")

    print("\nGESAMT:")
    print(f"  Gut        (≥ {_THRESH_GUT}):  {n_gut:,}  ({pct(n_gut)}%)")
    print(f"  Akzeptabel (≥ {_THRESH_AKZEPTABEL}):  {n_akzeptabel:,}  ({pct(n_akzeptabel)}%)")
    print(f"  Schlecht   (< {_THRESH_AKZEPTABEL}):  {n_schlecht:,}  ({pct(n_schlecht)}%)")

    print("\nScore-Statistik:")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Std:    {scores.std():.4f}")

    print("\nNACH QUELLE:")
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source]
        ns = len(sub)
        sg = int((sub["qualitaet"] == "gut").sum())
        sa = int((sub["qualitaet"] == "akzeptabel").sum())
        ss = int((sub["qualitaet"] == "schlecht").sum())
        pct_s = lambda k: f"{100.0 * k / ns:.1f}" if ns > 0 else "0.0"
        med = sub["labse_score"].median()
        print(f"  {source}:")
        print(
            f"    Gut: {sg} ({pct_s(sg)}%) | "
            f"Akzeptabel: {sa} ({pct_s(sa)}%) | "
            f"Schlecht: {ss} ({pct_s(ss)}%)"
        )
        print(f"    Median-Score: {med:.4f}")

    print("\nNACH LABEL (nur Positiv-Beispiele, label == 1.0):")
    for label in _LABELS:
        if label not in df.columns:
            continue
        pos = df[df[label] == 1.0]
        nl = len(pos)
        if nl == 0:
            print(f"  {label}=1:  n=0  (keine Positiv-Beispiele im Sample)")
            continue
        med_l = pos["labse_score"].median()
        bad_l = int((pos["qualitaet"] == "schlecht").sum())
        pct_bad_l = 100.0 * bad_l / nl
        print(
            f"  {label:<12}=1:  n={nl:<5}  "
            f"Median-Score: {med_l:.4f}  "
            f"Schlecht: {pct_bad_l:.1f}%"
        )

    print(f"\nHOCHRECHNUNG AUF GESAMTDATENSATZ:")
    print(f"  Silberdaten gesamt:  {total_silver_rows:,} Zeilen")
    print(f"  Erwartete schlechte Übersetzungen: {expected_bad:,} ({100.0 * bad_rate:.1f}%)")
    print(f"  Empfohlener Filter-Schwellwert: {_THRESH_AKZEPTABEL}")
    print(f"  Empfohlener Filterbefehl:")
    print(f'    df = df[df["labse_score"] >= {_THRESH_AKZEPTABEL}]')

    print(f"\n10 SCHLECHTESTE ÜBERSETZUNGEN:")
    print("─" * 34)
    worst = df.nsmallest(10, "labse_score")
    for i, (_, row) in enumerate(worst.iterrows(), start=1):
        label_str = "  ".join(
            f"{lbl}={int(row[lbl]) if not (isinstance(row[lbl], float) and np.isnan(row[lbl])) else 'NaN'}"
            for lbl in _LABELS
        )
        en_text = str(row["text_original"])[:120].replace("\n", " ")
        de_text = str(row["text"])[:120].replace("\n", " ")
        print(f"[{i}] Score: {row['labse_score']:.4f} | Quelle: {row['source']} | {label_str}")
        print(f'    EN: "{en_text}"')
        print(f'    DE: "{de_text}"')

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Function 6: save_excel
# ---------------------------------------------------------------------------

def save_excel(df: pd.DataFrame, output_path) -> None:
    """
    Save evaluation results to a 3-sheet Excel workbook.

    Sheet 1 — Alle_Samples:     all sampled rows sorted by score ascending
    Sheet 2 — Schlechte_Qualitaet: only schlecht rows
    Sheet 3 — Zusammenfassung:  pivot table + overall statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Restrict to Excel output columns (only include columns that exist)
    excel_cols = [c for c in _EXCEL_COLUMNS if c in df.columns]
    df_all = _sanitize_for_excel(
        df[excel_cols].sort_values("labse_score", ascending=True).reset_index(drop=True)
    )
    df_bad = df_all[df_all["qualitaet"] == "schlecht"]

    # Pivot: source × qualitaet
    pivot = (
        df.groupby(["source", "qualitaet"], observed=True)
        .size()
        .rename("Anzahl")
        .reset_index()
        .pivot(index="source", columns="qualitaet", values="Anzahl")
        .fillna(0)
        .astype(int)
    )
    # Ensure all three quality columns exist
    for col in ["gut", "akzeptabel", "schlecht"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["gut", "akzeptabel", "schlecht"]]
    pivot["gesamt"] = pivot.sum(axis=1)

    # Overall score statistics
    stats = df["labse_score"].describe().rename("labse_score")
    stats_df = stats.to_frame().T

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Alle_Samples",       index=False)
        df_bad.to_excel(writer, sheet_name="Schlechte_Qualitaet", index=False)

        # Write pivot and stats to Zusammenfassung sheet
        pivot.to_excel(writer, sheet_name="Zusammenfassung", startrow=0)
        stats_row = len(pivot) + 3
        stats_df.to_excel(
            writer, sheet_name="Zusammenfassung",
            startrow=stats_row, index=False,
        )
        ws = writer.sheets["Zusammenfassung"]
        ws.cell(row=stats_row, column=1, value="Score-Statistik:")


# ---------------------------------------------------------------------------
# Function 6b: save_large_results
# ---------------------------------------------------------------------------

def save_large_results(
    df: pd.DataFrame,
    parquet_path: Path,
    excel_path: Path,
) -> None:
    """
    For datasets larger than _LARGE_DATASET_THRESHOLD rows, save full scores to
    parquet and write a compact summary-only Excel (Worst_500 + pivot + stats).

    Args:
        df:           Scored and categorized DataFrame.
        parquet_path: Destination for the full-scores parquet file.
        excel_path:   Destination for the summary Excel file.
    """
    parquet_path = Path(parquet_path)
    excel_path = Path(excel_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_path, index=False)
    print(f"Parquet gespeichert: {parquet_path} ({len(df):,} Zeilen)")

    # Worst 500 rows for manual inspection
    excel_cols = [c for c in _EXCEL_COLUMNS if c in df.columns]
    worst_500 = _sanitize_for_excel(
        df.nsmallest(500, "labse_score")[excel_cols].reset_index(drop=True)
    )

    # Pivot: source × qualitaet
    pivot = (
        df.groupby(["source", "qualitaet"], observed=True)
        .size()
        .rename("Anzahl")
        .reset_index()
        .pivot(index="source", columns="qualitaet", values="Anzahl")
        .fillna(0)
        .astype(int)
    )
    for col in ["gut", "akzeptabel", "schlecht"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["gut", "akzeptabel", "schlecht"]]
    pivot["gesamt"] = pivot.sum(axis=1)

    stats_df = df["labse_score"].describe().rename("labse_score").to_frame().T

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        worst_500.to_excel(writer, sheet_name="Worst_500", index=False)
        pivot.to_excel(writer, sheet_name="Zusammenfassung", startrow=0)
        stats_row = len(pivot) + 3
        stats_df.to_excel(
            writer, sheet_name="Zusammenfassung",
            startrow=stats_row, index=False,
        )
        ws = writer.sheets["Zusammenfassung"]
        ws.cell(row=stats_row, column=1, value="Score-Statistik:")

    print(f"Excel-Zusammenfassung gespeichert: {excel_path}")


# ---------------------------------------------------------------------------
# Function 7: main
# ---------------------------------------------------------------------------

def main(
    sample_n: int = 1000,
    label: Optional[str] = None,
    positive_only: bool = False,
) -> None:
    """
    Full LaBSE translation quality evaluation pipeline.

    Args:
        sample_n:      Number of rows to sample for evaluation (default 1000).
        label:         Restrict to a specific label (one of _LABELS), or None for all.
        positive_only: When label is set, keep only positive examples (label == 1.0).
    """
    metadata_path = Path("data/processed/unified_with_quality_translated_with_metadata.parquet")

    # Derive output filename from label / positive_only flags
    if label is None:
        output_stem = "translation_quality_LaBSE"
    elif positive_only:
        output_stem = f"translation_quality_LaBSE_{label}_positive"
    else:
        output_stem = f"translation_quality_LaBSE_{label}_annotated"
    output_path = Path("data/evaluation") / f"{output_stem}.xlsx"

    if not metadata_path.exists():
        print(
            "FEHLER: Metadata-Datei nicht gefunden.\n"
            "Starte Pipeline mit:\n"
            "  python3 run_pipeline.py --translate --keep-translation-metadata"
        )
        sys.exit(1)

    df_all      = load_data(metadata_path)
    df_filtered = filter_by_label(df_all, label=label, positive_only=positive_only)
    df_sample   = sample_for_evaluation(df_filtered, n=sample_n)
    df_scored   = compute_labse_scores(df_sample)
    df_scored   = categorize_scores(df_scored)
    print_report(df_scored, total_silver_rows=len(df_filtered))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df_scored) > _LARGE_DATASET_THRESHOLD:
        parquet_out = Path("data/evaluation") / f"{output_stem}_all_scores.parquet"
        excel_out   = Path("data/evaluation") / f"{output_stem}_all_summary.xlsx"
        save_large_results(df_scored, parquet_out, excel_out)
    else:
        save_excel(df_scored, output_path)
        print(f"Gespeichert: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LaBSE translation quality evaluation for silver data.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=1000,
        help="Number of rows to sample for evaluation (default: 1000).",
    )
    parser.add_argument(
        "--label",
        choices=_LABELS,
        default=None,
        help="Restrict evaluation to rows annotated for this label.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="When --label is set, keep only positive examples (label == 1.0).",
    )
    args = parser.parse_args()
    main(
        sample_n=args.sample_n,
        label=args.label,
        positive_only=args.positive_only,
    )