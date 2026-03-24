"""
Loader for Jigsaw Multilingual Toxic Comment dataset.

Source:
    data/raw/jigsaw/train.csv        — training set with all labels
    data/raw/jigsaw/test.csv         — test texts only (no labels)
    data/raw/jigsaw/test_labels.csv  — test labels (-1 = not scored)

Labels provided (after schema-reduction merges):
    hate_speech  — derived from original identity_hate column
    toxic        — derived from original toxic | severe_toxic (OR-merge)
    threat       — kept as-is
    insult       — kept as-is

Merge rules applied:
    identity_hate → hate_speech
        Jigsaw's identity_hate annotates identity-group-targeted abuse,
        which is semantically a subtype of hate speech.
        Note: Jigsaw has no original "hate_speech" column; hate_speech here
        is sourced exclusively from identity_hate.

    severe_toxic | toxic → toxic
        severe_toxic is a strict subset of toxic in Jigsaw (every severely
        toxic post is also toxic). The OR-merge is a no-op in practice but
        is applied for semantic correctness.

    obscene → dropped
        Profanity is not reliably harmful and was annotated as a distinct
        category from toxicity. Dropped from the training schema.

Language: multilingual (treated as-is; corpus is mostly English)
Test rows with label == -1 (unscored) are dropped.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from ..base import BaseCorpusLoader

# Raw columns read from the Jigsaw CSV files.
# These are the original annotation columns, not the output schema columns.
_RAW_LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class JigsawLoader(BaseCorpusLoader):
    SOURCE = "jigsaw"
    LANGUAGE = "multilingual"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        jigsaw_dir = Path(data_dir) / "jigsaw"

        train = pd.read_csv(jigsaw_dir / "train.csv")
        train["split"] = "train"

        test_text   = pd.read_csv(jigsaw_dir / "test.csv")
        test_labels = pd.read_csv(jigsaw_dir / "test_labels.csv")
        test = test_text.merge(test_labels, on="id")
        test = test[test["toxic"] != -1]   # drop unscored rows
        test["split"] = "test"

        raw = pd.concat(
            [train[["comment_text", "split"] + _RAW_LABEL_COLS],
             test[["comment_text", "split"]  + _RAW_LABEL_COLS]],
            ignore_index=True,
        )

        return self._make_frame(
            texts=raw["comment_text"],
            splits=raw["split"],
            label_dict={
                # identity_hate → hate_speech (schema merge)
                "hate_speech": raw["identity_hate"].astype(float),
                # toxic | severe_toxic → toxic (OR-merge; severe_toxic ⊆ toxic in practice)
                "toxic":       (raw["toxic"].astype(bool) | raw["severe_toxic"].astype(bool)).astype(float),
                # threat: kept as-is
                "threat":      raw["threat"].astype(float),
                # insult: kept as-is (no attack column in Jigsaw)
                "insult":      raw["insult"].astype(float),
                # obscene: dropped from schema
            },
        )