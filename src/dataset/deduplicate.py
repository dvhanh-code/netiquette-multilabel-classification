"""
src/dataset/deduplicate.py
---------------------------
Remove exact duplicate texts from the unified dataset, keeping the
highest-quality representative row for each duplicated text.

Ranking priority (for ties broken top-to-bottom):
  1. Gold over silver  (is_gold: True > False)
  2. More positive labels  (sum of label == 1.0)
  3. More annotated labels (sum of label is not NaN)
  4. Longer text
  5. Lower original index  (stable, deterministic)

After deduplication, fresh train/val/test splits are assigned via
assign_fresh_splits():
  - Silver rows → train only
  - Gold rows   → stratified 70/15/15

Usage:
    python3 src/dataset/deduplicate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as `python3 src/dataset/deduplicate.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.splits import assign_fresh_splits

_LABELS     = ["hate_speech", "toxic", "threat", "insult"]
_INPUT_PATH = Path(
    "data/processed/unified_with_quality_translated_filtered_labse055.parquet"
)
_OUTPUT_PATH = Path("data/processed/unified_final_deduplicated.parquet")
_HELPER_COLS = [
    "_text_key", "_is_gold_int", "_pos_count", "_ann_count",
    "_text_len",  "_orig_idx",
]


# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_exact_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact-duplicate texts, keeping the best representative row.

    Ranking priority:
      1. Gold > silver
      2. More positive labels
      3. More annotated labels
      4. Longer text
      5. Lower original index (deterministic tiebreak)

    Returns a deduplicated copy; input is not mutated.
    """
    result = df.copy()

    # Build sort keys
    result["_text_key"] = result["text"].fillna("").astype(str).str.strip()
    result["_is_gold_int"] = result["is_gold"].astype(int)
    result["_pos_count"]   = (result[_LABELS] == 1.0).sum(axis=1)
    result["_ann_count"]   = result[_LABELS].notna().sum(axis=1)
    result["_text_len"] = result["text"].fillna("").astype(str).str.len()
    result["_orig_idx"]    = np.arange(len(result))

    # Sort so the best row for each text_key is first
    result = result.sort_values(
        by=[
            "_text_key",
            "_is_gold_int",   # descending: gold first
            "_pos_count",     # descending: more positives first
            "_ann_count",     # descending: more annotations first
            "_text_len",      # descending: longer text first
            "_orig_idx",      # ascending:  lower index first (tiebreak)
        ],
        ascending=[True, False, False, False, False, True],
    )

    result = result.drop_duplicates(subset=["_text_key"], keep="first")
    result = result.drop(columns=_HELPER_COLS).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_parquet(_INPUT_PATH)
    n_input = len(df)

    # ── Duplicate statistics ──────────────────────────────────────────────────
    key = df["text"].fillna("").astype(str).str.strip()
    dup_mask          = key.duplicated(keep=False)
    n_dup_rows        = int(dup_mask.sum())
    n_unique_dup_texts = int(key[dup_mask].nunique())

    print("=" * 60)
    print("DEDUPLICATION REPORT")
    print("=" * 60)
    print(f"  Input rows:              {n_input:,}")
    print(f"  Duplicate rows:          {n_dup_rows:,}")
    print(f"  Unique duplicated texts: {n_unique_dup_texts:,}")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    df_dedup = deduplicate_exact_texts(df)
    dedup_key = df_dedup["text"].fillna("").astype(str).str.strip()
    assert not dedup_key.duplicated().any(), \
        "Deduplication failed: duplicate text keys remain."
    n_removed = n_input - len(df_dedup)

    print(f"  Rows removed:            {n_removed:,}")
    print(f"  Output rows:             {len(df_dedup):,}")

    # ── Assign fresh splits ───────────────────────────────────────────────────
    df_final = assign_fresh_splits(df_dedup)

    # ── Split report ──────────────────────────────────────────────────────────
    print("\n  Overall split distribution:")
    for sp, cnt in df_final["split"].value_counts().items():
        print(f"    {sp:<6}: {cnt:,}")

    gold   = df_final[df_final["is_gold"] == True]
    silver = df_final[df_final["is_gold"] == False]

    print("\n  Gold split distribution:")
    for sp, cnt in gold["split"].value_counts().items():
        print(f"    {sp:<6}: {cnt:,}")

    print("\n  Silver split distribution:")
    for sp, cnt in silver["split"].value_counts().items():
        print(f"    {sp:<6}: {cnt:,}")

    print("\n  Gold threat=1 by split:")
    for sp in ["train", "val", "test"]:
        n = int((gold[gold["split"] == sp]["threat"] == 1.0).sum())
        print(f"    {sp:<6}: threat=1 → {n}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(_OUTPUT_PATH, index=False)
    print(f"\nSaved: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()