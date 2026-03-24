"""
Loader for HOCON34k: Hate speech in Online Comments from German Newspapers.

Source: data/raw/hocon34k/corpus-hocon34k/datasets/hatespeech_hocon34k.csv
Labels provided: hate_speech
Language: de

Multiple annotators per post are aggregated by majority vote (mean >= 0.5).
The split_all column is often empty; rows without a split value get split = None.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from ..base import BaseCorpusLoader


class HOCON34kLoader(BaseCorpusLoader):
    SOURCE = "hocon34k"
    LANGUAGE = "de"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        csv_path = (
            Path(data_dir)
            / "hocon34k"
            / "corpus-hocon34k"
            / "datasets"
            / "hatespeech_hocon34k.csv"
        )

        raw = pd.read_csv(csv_path)

        # Aggregate per post: majority vote across annotators
        agg = (
            raw.groupby("post_id", sort=False)
            .agg(text=("text", "first"), label_hs=("label_hs", "mean"), split_all=("split_all", "first"))
            .reset_index()
        )

        # Treat empty strings and NaN split values uniformly as None
        splits = agg["split_all"].where(
            agg["split_all"].notna() & (agg["split_all"].astype(str).str.strip() != ""),
            other=None,
        )

        return self._make_frame(
            texts=agg["text"],
            splits=splits,
            label_dict={"hate_speech": (agg["label_hs"] >= 0.5).astype(float)},
        )
