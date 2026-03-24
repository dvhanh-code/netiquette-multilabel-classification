"""
Loader for GMHP7k: German Misogynistic Hate Speech corpus.

Source: data/raw/corpus-gmhp7k/datasets/misogynistic_hatespeech_phase{1,2,3}.csv
Labels provided: hate_speech  (merged from original hs + m_hs columns)
Language: de

Merge rule applied:
    hate_speech = 1.0  if  (mean(hs) >= 0.5)  OR  (mean(m_hs) >= 0.5)
    i.e. a post is positive for hate_speech if a majority of annotators
    flagged it as either general hate speech or misogynistic hate speech.
    Information lost: the misogyny-specific signal is no longer separately
    available in the training schema.

Phase 3 includes split_hs / split_m_hs columns (train/test/val).
Phases 1 & 2 have no predefined split (split = None).
Multiple annotations per tweet are aggregated by majority vote (mean >= 0.5).
"""
from pathlib import Path
from typing import Union

import pandas as pd

from ..base import BaseCorpusLoader


class GMHP7kLoader(BaseCorpusLoader):
    SOURCE = "gmhp7k"
    LANGUAGE = "de"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        csv_dir = Path(data_dir) / "corpus-gmhp7k" / "datasets"

        parts = [
            pd.read_csv(p)
            for p in sorted(csv_dir.glob("misogynistic_hatespeech_phase*.csv"))
        ]
        raw = pd.concat(parts, ignore_index=True)

        # Aggregate per tweet: majority vote across annotators
        agg = (
            raw.groupby("tweet_id", sort=False)
            .agg(review_text=("review_text", "first"), hs=("hs", "mean"), m_hs=("m_hs", "mean"))
            .reset_index()
        )

        # Re-attach split from phase3 rows (other phases lack this column)
        if "split_hs" in raw.columns:
            split_map = (
                raw.dropna(subset=["split_hs"])
                .drop_duplicates("tweet_id")
                .set_index("tweet_id")["split_hs"]
            )
            agg["split"] = agg["tweet_id"].map(split_map)
        else:
            agg["split"] = None

        hate_speech = ((agg["hs"] >= 0.5) | (agg["m_hs"] >= 0.5)).astype(float)

        return self._make_frame(
            texts=agg["review_text"],
            splits=agg["split"],
            label_dict={"hate_speech": hate_speech},
        )
