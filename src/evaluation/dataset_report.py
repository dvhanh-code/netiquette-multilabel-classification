"""
src/evaluation/dataset_report.py
----------------------------------
Comprehensive dataset report for unified_final.parquet.
Generates console output, Excel workbook, and PNG figures.

Usage:
    python3 src/evaluation/dataset_report.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_LABELS      = ["hate_speech", "toxic", "threat", "insult"]
_GOLD_COLOR  = "#2E4057"
_SILVER_COLOR = "#02C39A"
_FIGURES_DIR = Path("data/evaluation/figures")
_EXCEL_PATH  = Path("data/evaluation/dataset_report.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lbl in _LABELS:
        col = df[lbl]
        annotated = col.notna().sum()
        pos = (col == 1.0).sum()
        neg = (col == 0.0).sum()
        nan = col.isna().sum()
        rate = pos / annotated if annotated > 0 else float("nan")
        rows.append({
            "Label":      lbl,
            "Annotiert":  int(annotated),
            "Positiv":    int(pos),
            "Negativ":    int(neg),
            "NaN":        int(nan),
            "Positiv-%":  f"{100 * rate:.1f}%" if not np.isnan(rate) else "—",
        })
    return pd.DataFrame(rows).set_index("Label")


def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n── {title} {'─' * pad}")
    else:
        print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Basic statistics
# ─────────────────────────────────────────────────────────────────────────────

def section_1_basic_stats(df: pd.DataFrame) -> None:
    _sep("1. GRUNDSTATISTIK")

    n_total  = len(df)
    n_gold   = int(df["is_gold"].sum())
    n_silver = n_total - n_gold

    print(f"  Gesamt:          {n_total:,}")
    print(f"  Gold:            {n_gold:,}  ({100*n_gold/n_total:.1f}%)")
    print(f"  Silver:          {n_silver:,}  ({100*n_silver/n_total:.1f}%)")

    _sep()
    print("  Split-Verteilung:")
    for sp, cnt in df["split"].value_counts().items():
        print(f"    {sp:<6}: {cnt:,}  ({100*cnt/n_total:.1f}%)")

    _sep()
    print("  Quellen:")
    for src, cnt in df["source"].value_counts().items():
        gold_cnt = int(df[(df["source"] == src) & df["is_gold"]].shape[0])
        print(f"    {src:<20}: {cnt:,}  (Gold: {gold_cnt:,})")

    _sep()
    print("  Sprachen:")
    for lang, cnt in df["language"].value_counts().items():
        print(f"    {lang}: {cnt:,}")

    _sep()
    dups = int(df["text"].duplicated().sum())
    missing_text = int(df["text"].isna().sum())
    print(f"  Exakte Duplikate (Text):   {dups:,}")
    print(f"  Fehlende Texte:            {missing_text:,}")

    _sep()
    print("  Ø Textlänge (Zeichen) pro Quelle:")
    for src in df["source"].unique():
        mean_len = df.loc[df["source"] == src, "text"].str.len().mean()
        print(f"    {src:<20}: {mean_len:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Label distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_2_label_distribution(df: pd.DataFrame) -> None:
    _sep("2. LABEL-VERTEILUNG")

    gold   = df[df["is_gold"] == True]
    silver = df[df["is_gold"] == False]

    print("\n  a) Gesamtdatensatz:")
    print(_label_stats(df).to_string())

    print("\n  b) Gold:")
    print(_label_stats(gold).to_string())

    print("\n  c) Silver:")
    print(_label_stats(silver).to_string())

    print("\n  d) Gold nach Split:")
    for sp in ["train", "val", "test"]:
        sub = gold[gold["split"] == sp]
        print(f"\n     Gold {sp} (n={len(sub):,}):")
        print(_label_stats(sub).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Label co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

def section_3_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    _sep("3. LABEL-KOOKKURRENZ (Positiv-Beispiele)")

    pos = df[_LABELS].fillna(0)

    # Co-occurrence: count rows where both label A and B are 1
    cooc = pd.DataFrame(index=_LABELS, columns=_LABELS, dtype=int)
    for a in _LABELS:
        for b in _LABELS:
            cooc.loc[a, b] = int(((pos[a] == 1) & (pos[b] == 1)).sum())

    print("\n  Kookkurrenz-Matrix (Anzahl Zeilen mit beiden Labels == 1):")
    print(cooc.to_string())

    # Jaccard similarity
    jacc = pd.DataFrame(index=_LABELS, columns=_LABELS, dtype=float)
    for a in _LABELS:
        for b in _LABELS:
            inter = int(((pos[a] == 1) & (pos[b] == 1)).sum())
            union = int(((pos[a] == 1) | (pos[b] == 1)).sum())
            jacc.loc[a, b] = inter / union if union > 0 else 0.0

    print("\n  Jaccard-Ähnlichkeit:")
    print(jacc.round(3).to_string())

    return cooc


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Text length
# ─────────────────────────────────────────────────────────────────────────────

def section_4_text_length(df: pd.DataFrame) -> None:
    _sep("4. TEXTLÄNGEN-ANALYSE")

    df = df.copy()
    df["text_len"] = df["text"].str.len()

    print(f"\n  {'Split':<8}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  "
          f"{'Min':>6}  {'Max':>6}  {'<50':>6}  {'>500':>6}")
    print("  " + "─" * 66)

    for sp in ["train", "val", "test"]:
        sub = df.loc[df["split"] == sp, "text_len"]
        pct_short = 100 * (sub < 50).sum() / len(sub)
        pct_long  = 100 * (sub > 500).sum() / len(sub)
        print(
            f"  {sp:<8}  {sub.mean():>7.0f}  {sub.median():>7.0f}  "
            f"{sub.std():>7.0f}  {sub.min():>6}  {sub.max():>6}  "
            f"{pct_short:>5.1f}%  {pct_long:>5.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: LaBSE quality
# ─────────────────────────────────────────────────────────────────────────────

def section_5_labse_quality(df: pd.DataFrame) -> None:
    _sep("5. LABSE-QUALITÄT (Silver-Daten)")

    silver = df[df["is_gold"] == False].copy()
    scores = silver["labse_score"].dropna()

    print(f"\n  Gesamt Silver-Zeilen mit LaBSE-Score: {len(scores):,}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")

    print("\n  Verteilung nach Quelle:")
    for src in silver["source"].unique():
        sub = silver.loc[silver["source"] == src, "labse_score"].dropna()
        if len(sub) == 0:
            continue
        print(f"    {src:<22}: n={len(sub):,}  "
              f"mean={sub.mean():.4f}  median={sub.median():.4f}")

    print("\n  Verteilung nach Label (Positiv-Beispiele):")
    for lbl in _LABELS:
        sub = silver.loc[silver[lbl] == 1.0, "labse_score"].dropna()
        if len(sub) == 0:
            print(f"    {lbl:<12}: keine Positiv-Beispiele in Silver")
            continue
        print(f"    {lbl:<12}: n={len(sub):,}  "
              f"mean={sub.mean():.4f}  median={sub.median():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Excel report
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    import re
    illegal = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].apply(
            lambda v: illegal.sub("", v) if isinstance(v, str) else v
        )
    return out


def section_6_excel(df: pd.DataFrame) -> None:
    _EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    gold   = df[df["is_gold"] == True]
    silver = df[df["is_gold"] == False]

    # ── Sheet 1: Übersicht ────────────────────────────────────────────────────
    n_total = len(df)
    n_gold  = len(gold)
    overview_rows = [
        ("Gesamt", n_total),
        ("Gold", n_gold),
        ("Silver", n_total - n_gold),
        ("Gold-Anteil", f"{100*n_gold/n_total:.1f}%"),
        ("Sprachen", ", ".join(df["language"].unique())),
        ("Quellen", ", ".join(sorted(df["source"].unique()))),
        ("Duplikate (Text)", int(df["text"].duplicated().sum())),
        ("Fehlende Texte", int(df["text"].isna().sum())),
    ]
    for sp in ["train", "val", "test"]:
        cnt = int((df["split"] == sp).sum())
        overview_rows.append((f"Split: {sp}", cnt))

    df_overview = pd.DataFrame(overview_rows, columns=["Kennzahl", "Wert"])

    # ── Sheet 2: Label-Verteilung ─────────────────────────────────────────────
    stats_full   = _label_stats(df).reset_index()
    stats_gold   = _label_stats(gold).reset_index().add_prefix("Gold_")
    stats_silver = _label_stats(silver).reset_index().add_prefix("Silver_")
    stats_full.columns   = ["Label"] + [f"Gesamt_{c}" for c in stats_full.columns[1:]]
    df_labels = pd.concat(
        [stats_full, stats_gold.rename(columns={"Gold_Label": "Label"}),
         stats_silver.rename(columns={"Silver_Label": "Label"})],
        axis=1
    )
    # Keep one Label column
    df_labels = df_labels.loc[:, ~df_labels.columns.duplicated()]

    # ── Sheet 3: Split-Verteilung ─────────────────────────────────────────────
    split_rows = []
    for src in sorted(df["source"].unique()):
        for gtype, sub in [("Gold", gold), ("Silver", silver)]:
            sub_src = sub[sub["source"] == src]
            row = {"Quelle": src, "Typ": gtype}
            for sp in ["train", "val", "test"]:
                row[sp] = int((sub_src["split"] == sp).sum())
            row["Gesamt"] = len(sub_src)
            split_rows.append(row)
    df_splits = pd.DataFrame(split_rows)

    # ── Sheet 4: Kookkurrenz ──────────────────────────────────────────────────
    pos = df[_LABELS].fillna(0)
    cooc_data = {}
    for a in _LABELS:
        cooc_data[a] = {
            b: int(((pos[a] == 1) & (pos[b] == 1)).sum())
            for b in _LABELS
        }
    df_cooc = pd.DataFrame(cooc_data).T

    jacc_data = {}
    for a in _LABELS:
        jacc_data[a] = {}
        for b in _LABELS:
            inter = int(((pos[a] == 1) & (pos[b] == 1)).sum())
            union = int(((pos[a] == 1) | (pos[b] == 1)).sum())
            jacc_data[a][b] = round(inter / union, 4) if union > 0 else 0.0
    df_jacc = pd.DataFrame(jacc_data).T

    # ── Sheet 5: Textlängen ───────────────────────────────────────────────────
    df_copy = df.copy()
    df_copy["text_len"] = df_copy["text"].str.len()
    tlen_rows = []
    for src in sorted(df_copy["source"].unique()):
        for sp in ["train", "val", "test"]:
            sub = df_copy.loc[
                (df_copy["source"] == src) & (df_copy["split"] == sp),
                "text_len",
            ]
            if len(sub) == 0:
                continue
            tlen_rows.append({
                "Quelle": src, "Split": sp,
                "n":      len(sub),
                "Mean":   round(sub.mean(), 1),
                "Median": round(sub.median(), 1),
                "Std":    round(sub.std(), 1),
                "Min":    int(sub.min()),
                "Max":    int(sub.max()),
                "<50_Pct":  round(100 * (sub < 50).sum() / len(sub), 1),
                ">500_Pct": round(100 * (sub > 500).sum() / len(sub), 1),
            })
    df_tlen = pd.DataFrame(tlen_rows)

    # ── Sheet 6: LaBSE-Qualität ───────────────────────────────────────────────
    silver_s = df[df["is_gold"] == False].copy()
    labse_rows = []
    for src in sorted(silver_s["source"].unique()):
        sub = silver_s.loc[silver_s["source"] == src, "labse_score"].dropna()
        if len(sub) == 0:
            continue
        labse_rows.append({
            "Quelle": src, "Label": "—",
            "n": len(sub),
            "Mean": round(sub.mean(), 4),
            "Median": round(sub.median(), 4),
            "Std": round(sub.std(), 4),
        })
    for lbl in _LABELS:
        sub = silver_s.loc[silver_s[lbl] == 1.0, "labse_score"].dropna()
        if len(sub) == 0:
            continue
        labse_rows.append({
            "Quelle": "— (Positiv)", "Label": lbl,
            "n": len(sub),
            "Mean": round(sub.mean(), 4),
            "Median": round(sub.median(), 4),
            "Std": round(sub.std(), 4),
        })
    df_labse = pd.DataFrame(labse_rows)

    # ── Write workbook ────────────────────────────────────────────────────────
    with pd.ExcelWriter(_EXCEL_PATH, engine="openpyxl") as writer:
        df_overview.to_excel(writer, sheet_name="Übersicht",       index=False)
        df_labels.to_excel(  writer, sheet_name="Label_Verteilung", index=False)
        df_splits.to_excel(  writer, sheet_name="Split_Verteilung", index=False)

        df_cooc.to_excel(writer, sheet_name="Kookkurrenz", startrow=0)
        df_jacc.to_excel(writer, sheet_name="Kookkurrenz", startrow=len(df_cooc) + 3)
        ws = writer.sheets["Kookkurrenz"]
        ws.cell(row=1, column=1, value="Kookkurrenz-Matrix")
        ws.cell(row=len(df_cooc) + 4, column=1, value="Jaccard-Ähnlichkeit")

        df_tlen.to_excel( writer, sheet_name="Textlaengen",   index=False)
        df_labse.to_excel(writer, sheet_name="LaBSE_Qualitaet", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Figures
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(name: str) -> None:
    plt.tight_layout()
    path = _FIGURES_DIR / name
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Gespeichert: {path}")


def section_7_figures(df: pd.DataFrame) -> None:
    _sep("7. ABBILDUNGEN")
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gold   = df[df["is_gold"] == True]
    silver = df[df["is_gold"] == False]

    # ── Figure 1: Label distribution Gold vs Silver ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(_LABELS))
    w = 0.35
    gold_pos   = [(gold[lbl] == 1.0).sum() for lbl in _LABELS]
    silver_pos = [(silver[lbl] == 1.0).sum() for lbl in _LABELS]

    ax.bar(x - w/2, gold_pos,   w, label="Gold",   color=_GOLD_COLOR)
    ax.bar(x + w/2, silver_pos, w, label="Silver", color=_SILVER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS)
    ax.set_xlabel("Label")
    ax.set_ylabel("Anzahl Positiv-Beispiele")
    ax.set_title("Label-Verteilung: Gold vs. Silver")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _savefig("label_distribution.png")

    # ── Figure 2: Split distribution stacked Gold/Silver ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    splits = ["train", "val", "test"]
    gold_counts   = [(gold["split"] == sp).sum()   for sp in splits]
    silver_counts = [(silver["split"] == sp).sum() for sp in splits]

    ax.bar(splits, gold_counts,   label="Gold",   color=_GOLD_COLOR)
    ax.bar(splits, silver_counts, bottom=gold_counts, label="Silver", color=_SILVER_COLOR)
    ax.set_xlabel("Split")
    ax.set_ylabel("Anzahl Zeilen")
    ax.set_title("Split-Verteilung")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _savefig("split_distribution.png")

    # ── Figure 3: Co-occurrence heatmap ──────────────────────────────────────
    pos = df[_LABELS].fillna(0)
    cooc = np.array([
        [int(((pos[a] == 1) & (pos[b] == 1)).sum()) for b in _LABELS]
        for a in _LABELS
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cooc, annot=True, fmt="d", cmap="Blues",
        xticklabels=_LABELS, yticklabels=_LABELS, ax=ax,
    )
    ax.set_title("Label-Kookkurrenz (Positiv-Beispiele)")
    _savefig("cooccurrence_heatmap.png")

    # ── Figure 4: Text length distribution Gold vs Silver ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1000, 50)
    gold_lens   = gold["text"].str.len().clip(upper=1000)
    silver_lens = silver["text"].str.len().clip(upper=1000)
    ax.hist(gold_lens,   bins=bins, alpha=0.6, label="Gold",   color=_GOLD_COLOR,   density=True)
    ax.hist(silver_lens, bins=bins, alpha=0.6, label="Silver", color=_SILVER_COLOR, density=True)
    ax.set_xlabel("Textlänge (Zeichen, max. 1000)")
    ax.set_ylabel("Dichte")
    ax.set_title("Textlängenverteilung")
    ax.legend()
    _savefig("text_length_distribution.png")

    # ── Figure 5: LaBSE score distribution by source ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    silver_valid = silver[silver["labse_score"].notna()]
    palette = sns.color_palette("Set2", n_colors=silver_valid["source"].nunique())
    for i, src in enumerate(sorted(silver_valid["source"].unique())):
        sub = silver_valid.loc[silver_valid["source"] == src, "labse_score"]
        ax.hist(sub, bins=50, alpha=0.5, label=src, color=palette[i], density=True)
    ax.axvline(0.55, color="red", linestyle="--", linewidth=1.5, label="Schwellwert 0.55")
    ax.set_xlabel("LaBSE Score")
    ax.set_ylabel("Dichte")
    ax.set_title("LaBSE Score-Verteilung (Silver Data)")
    ax.legend(fontsize=8)
    _savefig("labse_score_distribution.png")

    # ── Figure 6: Label per split (Gold only) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    n_labels = len(_LABELS)
    n_splits = 3
    splits   = ["train", "val", "test"]
    x = np.arange(n_labels)
    w = 0.25
    colors = ["#2E4057", "#048A81", "#54C6EB"]

    for i, sp in enumerate(splits):
        sub = gold[gold["split"] == sp]
        counts = [(sub[lbl] == 1.0).sum() for lbl in _LABELS]
        ax.bar(x + (i - 1) * w, counts, w, label=sp, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS)
    ax.set_xlabel("Label")
    ax.set_ylabel("Anzahl Positiv-Beispiele")
    ax.set_title("Label-Verteilung pro Split (Gold)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _savefig("label_per_split.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_parquet("data/final/unified_final_v1.parquet")

    print("=" * 60)
    print("DATASET REPORT — unified_final.parquet")
    print("=" * 60)

    section_1_basic_stats(df)
    section_2_label_distribution(df)
    cooc = section_3_cooccurrence(df)
    section_4_text_length(df)
    section_5_labse_quality(df)
    section_6_excel(df)
    section_7_figures(df)

    print("\nReport saved:")
    print(f"  {_EXCEL_PATH}")
    print(f"  {_FIGURES_DIR}/*.png")


if __name__ == "__main__":
    main()
