"""
Loader for Detox: Wikipedia Toxicity dataset.

Source:
    data/raw/detox/toxicity_comments.tsv   — one of: text or annotations
    data/raw/detox/toxicity_annotations.tsv — one of: annotations or text

Note: the two filenames are counterintuitively swapped in this corpus release.
The loader detects which file contains text (column "comment") at runtime.

Labels provided: toxic
Language: en

Multiple workers per comment are aggregated by majority vote (mean toxicity >= 0.5).
NEWLINE_TOKEN placeholders are replaced with actual newlines.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from ..base import BaseCorpusLoader


class DetoxLoader(BaseCorpusLoader):
    SOURCE = "detox"
    LANGUAGE = "en"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        detox_dir = Path(data_dir) / "detox"

        f1 = pd.read_csv(detox_dir / "toxicity_comments.tsv", sep="\t")
        f2 = pd.read_csv(detox_dir / "toxicity_annotations.tsv", sep="\t")

        # Identify files by their columns — filenames are misleadingly swapped
        if "comment" in f1.columns:
            comments, annotations = f1, f2
        else:
            comments, annotations = f2, f1

        # Aggregate toxicity per comment (majority vote)
        agg = (
            annotations.groupby("rev_id")["toxicity"]
            .mean()
            .reset_index()
            .rename(columns={"toxicity": "toxic_score"})
        )
        agg["toxic"] = (agg["toxic_score"] >= 0.5).astype(float)

        merged = comments.merge(agg[["rev_id", "toxic"]], on="rev_id")
        merged["comment"] = merged["comment"].str.replace("NEWLINE_TOKEN", "\n", regex=False)

        splits = merged["split"] if "split" in merged.columns else None

        return self._make_frame(
            texts=merged["comment"],
            splits=splits,
            label_dict={"toxic": merged["toxic"]},
        )
