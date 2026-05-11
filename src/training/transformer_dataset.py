"""
src/training/transformer_dataset.py
-----------------------------------
Dataset utilities for Transformer-based multilabel classification.

This module handles partially labeled multilabel data:
    - NaN labels are replaced by 0.0 in the label tensor
    - label_mask marks which labels are actually annotated

Example:
    Original labels:
        hate_speech = NaN
        toxic       = 1.0
        threat      = NaN
        insult      = 0.0

    Tensor output:
        labels     = [0.0, 1.0, 0.0, 0.0]
        label_mask = [0.0, 1.0, 0.0, 1.0]
"""

from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


LABELS = ["hate_speech", "toxic", "threat", "insult"]


class NetiquetteTransformerDataset(Dataset):
    """
    PyTorch Dataset for partially labeled multilabel text classification.

    Each item contains:
        input_ids
        attention_mask
        labels
        label_mask
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        labels: Optional[List[str]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels if labels is not None else LABELS

        required_cols = ["text"] + self.labels
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.texts = self.df["text"].fillna("").astype(str).tolist()

        label_df = self.df[self.labels]

        # label_mask: 1 where label is annotated, 0 where NaN
        self.label_mask = (~label_df.isna()).astype("float32").values

        # labels: replace NaN by 0.0, but these positions will be ignored by mask
        self.label_values = label_df.fillna(0.0).astype("float32").values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_values[idx], dtype=torch.float),
            "label_mask": torch.tensor(self.label_mask[idx], dtype=torch.float),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


def load_dataset(
    path: str,
    mode: str = "gold_only",
) -> Dict[str, pd.DataFrame]:
    """
    Load frozen dataset and return train/val/test DataFrames.

    Args:
        path:
            Path to unified_final_v1.parquet.
        mode:
            "gold_only":
                train = gold train only
                val   = gold val
                test  = gold test

            "gold_silver":
                train = all train rows, including gold + silver
                val   = gold val only
                test  = gold test only

    Returns:
        Dictionary with keys: train, val, test
    """
    df = pd.read_parquet(path)

    if mode not in {"gold_only", "gold_silver"}:
        raise ValueError("mode must be either 'gold_only' or 'gold_silver'")

    if mode == "gold_only":
        train_df = df[(df["split"] == "train") & (df["is_gold"] == True)].copy()
    else:
        train_df = df[df["split"] == "train"].copy()

    val_df = df[(df["split"] == "val") & (df["is_gold"] == True)].copy()
    test_df = df[(df["split"] == "test") & (df["is_gold"] == True)].copy()

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def print_dataset_summary(splits: Dict[str, pd.DataFrame]) -> None:
    """Print compact summary for train/val/test DataFrames."""
    print("\nDataset summary:")
    for split_name, split_df in splits.items():
        gold_count = int((split_df["is_gold"] == True).sum())
        silver_count = int((split_df["is_gold"] == False).sum())

        print(f"\n{split_name.upper()}:")
        print(f"  rows   : {len(split_df):,}")
        print(f"  gold   : {gold_count:,}")
        print(f"  silver : {silver_count:,}")

        for label in LABELS:
            annotated = int(split_df[label].notna().sum())
            positive = int((split_df[label] == 1.0).sum())
            negative = int((split_df[label] == 0.0).sum())
            print(
                f"  {label:<12} annotated={annotated:>7,} "
                f"pos={positive:>6,} neg={negative:>7,}"
            )


if __name__ == "__main__":
    DATA_PATH = "data/final/unified_final_v1.parquet"

    for mode in ["gold_only", "gold_silver"]:
        print("=" * 80)
        print(f"MODE: {mode}")
        print("=" * 80)
        splits = load_dataset(DATA_PATH, mode=mode)
        print_dataset_summary(splits)