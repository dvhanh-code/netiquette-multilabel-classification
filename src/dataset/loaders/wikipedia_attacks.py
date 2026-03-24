"""
Loader for Wikipedia Personal Attacks dataset.

Source:
    data/raw/wikipedia_attacks/attack_annotated_comments.csv
    data/raw/wikipedia_attacks/attack_annotations.csv

Labels provided: insult  (merged from original attack annotation)
Language: en

Merge rule applied:
    attack → insult
        The original corpus annotates interpersonal attacks (targeted personal
        attacks against a specific user). Under the reduced schema, this is
        reported as insult, which covers both insulting language and personal
        targeting. Information lost: the attack-vs-insult distinction is no
        longer separately recoverable from the training schema.

Multiple workers per comment are aggregated by majority vote (mean attack >= 0.5).
NEWLINE_TOKEN placeholders are replaced with actual newlines.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from ..base import BaseCorpusLoader


class WikipediaAttacksLoader(BaseCorpusLoader):
    SOURCE = "wikipedia_attacks"
    LANGUAGE = "en"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        attacks_dir = Path(data_dir) / "wikipedia_attacks"

        comments = pd.read_csv(attacks_dir / "attack_annotated_comments.csv")
        annotations = pd.read_csv(attacks_dir / "attack_annotations.csv")

        # Aggregate attack label per comment (majority vote)
        agg = (
            annotations.groupby("rev_id")["attack"]
            .mean()
            .reset_index()
            .rename(columns={"attack": "attack_score"})
        )
        agg["insult"] = (agg["attack_score"] >= 0.5).astype(float)

        comments["comment"] = comments["comment"].str.replace(
            "NEWLINE_TOKEN", "\n", regex=False
        )

        merged = comments.merge(agg[["rev_id", "insult"]], on="rev_id")
        splits = merged["split"] if "split" in merged.columns else None

        return self._make_frame(
            texts=merged["comment"],
            splits=splits,
            label_dict={"insult": merged["insult"]},
        )
