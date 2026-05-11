from pathlib import Path

import numpy as np
import pandas as pd

from ..base import BaseCorpusLoader
from ..schema import SCHEMA_COLUMNS


class RPModLoader(BaseCorpusLoader):
    """
    Loader for RP-Mod & RP-Crowd dataset.

    Source: Assenmacher et al. (2021)
            NeurIPS Datasets and Benchmarks
            DOI: 10.5281/zenodo.5291339

    Uses RP-Mod-Crowd.csv (85,000 rows total).
    Only the 28,833 crowd-annotated rows are used
    (rows where Threat Count Crowd is not NaN).

    5 crowd annotators per comment.
    Threshold: >= 2 annotators = positive label.

    Merge rules:
      Racism Count Crowd >= 2
        OR Sexism Count Crowd >= 2  → hate_speech = 1.0
      Threat Count Crowd >= 2       → threat      = 1.0
      Insult Count Crowd >= 2       → insult      = 1.0
      toxic                         → NaN always
        (Profanity dropped like obscene in Jigsaw:
         profanity != reliably harmful)
    """

    SOURCE   = "rp_mod"
    LANGUAGE = "de"

    def load(self, data_dir) -> pd.DataFrame:
        path = Path(data_dir) / "rp_mod" / "RP-Mod-Crowd.csv"

        df = pd.read_csv(path)

        # Only crowd-annotated rows
        df = df[df["Threat Count Crowd"].notna()].copy()

        # Clean text
        df["text_clean"] = df["Text"].fillna("").str.strip()
        df = df[df["text_clean"].str.len() > 0]

        THRESH = 2

        # hate_speech: Racism OR Sexism >= threshold
        hate = (
            (df["Racism Count Crowd"] >= THRESH) |
            (df["Sexism Count Crowd"] >= THRESH)
        ).astype(float)

        # threat
        threat = (df["Threat Count Crowd"] >= THRESH).astype(float)

        # insult
        insult = (df["Insult Count Crowd"] >= THRESH).astype(float)

        # toxic = NaN always (not annotated by RP-Mod)
        toxic = pd.Series(np.nan, index=df.index)

        texts  = df["text_clean"].tolist()
        splits = [None] * len(df)

        label_dict = {
            "hate_speech": hate.tolist(),
            "toxic":       toxic.tolist(),
            "threat":      threat.tolist(),
            "insult":      insult.tolist(),
        }

        return self._make_frame(texts, splits, label_dict)
