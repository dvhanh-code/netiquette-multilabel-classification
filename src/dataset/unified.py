"""
UnifiedCorpusDataset — combines all six corpora into a single DataFrame
with a consistent label schema.

Usage
-----
    from src.dataset.unified import UnifiedCorpusDataset

    dataset = UnifiedCorpusDataset("data/raw")
    df = dataset.load()                      # load all corpora
    df = dataset.load(["gmhp7k", "jigsaw"]) # load a subset

    dataset.label_stats()                    # label distribution per corpus
    dataset.get_split("train")               # filter by split
    dataset.get_corpus("jigsaw")             # filter by source
"""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .loaders import ALL_LOADERS
from .schema import ALL_LABELS, CORPUS_LABELS


class UnifiedCorpusDataset:
    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir)
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, corpora: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and concatenate corpora.

        Args:
            corpora: list of corpus names to load, e.g. ["gmhp7k", "jigsaw"].
                     Loads all six corpora when None.

        Returns:
            Unified DataFrame with columns:
                text, source, language, split,
                hate_speech, misogyny, toxic, severe_toxic, obscene,
                threat, insult, identity_hate, attack, impolite
        """
        if corpora is None:
            loaders = ALL_LOADERS
        else:
            unknown = set(corpora) - set(ALL_LOADERS)
            if unknown:
                raise ValueError(f"Unknown corpora: {unknown}. Available: {list(ALL_LOADERS)}")
            loaders = {k: ALL_LOADERS[k] for k in corpora}

        frames = []
        for name, loader_cls in loaders.items():
            print(f"  Loading {name}...", end=" ", flush=True)
            df = loader_cls().load(self.data_dir)
            frames.append(df)
            print(f"{len(df):,} rows")

        self._df = pd.concat(frames, ignore_index=True)
        print(f"  Total: {len(self._df):,} rows across {len(frames)} corpus/corpora.")
        return self._df

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .load() first.")
        return self._df

    def get_split(self, split: str) -> pd.DataFrame:
        """Return rows matching the given split ('train', 'test', 'val')."""
        return self.df[self.df["split"] == split].reset_index(drop=True)

    def get_corpus(self, source: str) -> pd.DataFrame:
        """Return rows from a single corpus by source name."""
        return self.df[self.df["source"] == source].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def label_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame with per-label statistics:
            total     — number of annotated rows (non-NaN)
            positive  — number of positive (1.0) rows
            negative  — number of negative (0.0) rows
            pos_rate  — positive / total
        """
        rows = []
        for label in ALL_LABELS:
            col = self.df[label]
            total = int(col.notna().sum())
            positive = int((col == 1.0).sum())
            negative = int((col == 0.0).sum())
            rows.append({
                "label":    label,
                "corpora":  ", ".join(
                    c for c, labels in CORPUS_LABELS.items() if label in labels
                ),
                "total":    total,
                "positive": positive,
                "negative": negative,
                "pos_rate": round(positive / total, 4) if total > 0 else float("nan"),
            })
        return pd.DataFrame(rows).set_index("label")

    def split_stats(self) -> pd.DataFrame:
        """Return row counts per corpus × split."""
        return (
            self.df.groupby(["source", "split"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["source", "split"])
        )
