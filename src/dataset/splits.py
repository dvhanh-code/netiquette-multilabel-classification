"""
src/dataset/splits.py
----------------------
Assigns fresh train/val/test splits to a unified multilabel DataFrame.

Silver rows (is_gold=False) are always assigned to "train".
Gold rows (is_gold=True) are split 70/15/15 using multilabel stratification.
All pre-existing split values in the input are ignored.
"""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

_STRATIFY_LABELS = ["hate_speech", "toxic", "threat", "insult"]


def assign_fresh_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_state: int  = 42,
) -> pd.DataFrame:
    """
    Assign fresh train/val/test splits, ignoring any pre-existing split column.

    Silver rows (is_gold=False) → "train" always.
    Gold rows (is_gold=True)    → stratified 70/15/15 split.

    NaN labels are treated as 0 only for stratification; the returned
    DataFrame preserves original NaN values in all label columns.

    Returns a copy; input is never mutated.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    result = df.copy()

    # Wipe all existing split values — fresh assignment for every row
    result["split"] = None

    # ── Silver rows → train ───────────────────────────────────────────────────
    silver_mask = result["is_gold"] == False
    result.loc[silver_mask, "split"] = "train"
    n_silver = int(silver_mask.sum())

    # ── Gold rows → stratified 70/15/15 ──────────────────────────────────────
    gold_mask = result["is_gold"] == True
    gold_idx  = result.index[gold_mask]
    n_gold    = len(gold_idx)

    if n_gold > 0:
        gold_sub = result.loc[gold_idx]
        X = gold_sub[_STRATIFY_LABELS].fillna(0).values.astype(float)

        try:
            # Step 1: split off test (test_ratio of total)
            msss1 = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio,
                random_state=random_state,
            )
            trainval_pos, test_pos = next(msss1.split(X, X))

            # Step 2: from train+val portion, split off val
            val_ratio_of_trainval = val_ratio / (train_ratio + val_ratio)
            X_trainval = X[trainval_pos]
            msss2 = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio_of_trainval,
                random_state=random_state,
            )
            train_pos2, val_pos2 = next(msss2.split(X_trainval, X_trainval))

            gold_positions = gold_idx.to_numpy()
            train_idx = gold_positions[trainval_pos[train_pos2]]
            val_idx   = gold_positions[trainval_pos[val_pos2]]
            test_idx  = gold_positions[test_pos]

        except Exception as e:
            print(f"Warning: multilabel stratification failed ({e}). "
                  "Falling back to random split.")
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n_gold)
            gold_positions = gold_idx.to_numpy()
            n_test = round(test_ratio * n_gold)
            n_val  = round(val_ratio  * n_gold)
            test_idx  = gold_positions[perm[:n_test]]
            val_idx   = gold_positions[perm[n_test:n_test + n_val]]
            train_idx = gold_positions[perm[n_test + n_val:]]

        result.loc[train_idx, "split"] = "train"
        result.loc[val_idx,   "split"] = "val"
        result.loc[test_idx,  "split"] = "test"

    # ── Assertions ────────────────────────────────────────────────────────────
    assert result["split"].isna().sum() == 0, \
        "BUG: some rows have no split assigned"

    all_splits = set(result["split"].unique())
    assert all_splits <= {"train", "val", "test"}, \
        f"BUG: unexpected split values: {all_splits - {'train', 'val', 'test'}}"

    silver_not_train = (
        (result["is_gold"] == False) & (result["split"] != "train")
    ).sum()
    assert silver_not_train == 0, \
        f"BUG: {silver_not_train:,} silver rows not in train"

    gold_splits = set(result.loc[result["is_gold"] == True, "split"].unique())
    assert {"train", "val", "test"}.issubset(gold_splits), \
        f"BUG: not all 3 splits present in gold rows. Found: {gold_splits}"

    # ── Summary ───────────────────────────────────────────────────────────────
    gold_result = result[result["is_gold"] == True]
    print(f"\nSplit assignment summary:")
    print(f"  Total rows:              {len(result):,}")
    print(f"  Silver → train:          {n_silver:,}")
    print(f"  Gold total:              {n_gold:,}")
    for sp in ["train", "val", "test"]:
        n = int((gold_result["split"] == sp).sum())
        print(f"    gold {sp:<5}:           {n:,}")
    print(f"  Gold threat=1 by split:")
    for sp in ["train", "val", "test"]:
        sub = gold_result[gold_result["split"] == sp]
        n = int((sub["threat"] == 1.0).sum())
        print(f"    {sp:<5}: threat=1 → {n}")

    return result


def assign_missing_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_state: int  = 42,
) -> pd.DataFrame:
    """Alias for assign_fresh_splits (backwards compatibility)."""
    return assign_fresh_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )